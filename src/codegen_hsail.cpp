#if HAS_HSA

#include "libHSAIL/HSAILScanner.h"

#define HSA_IR 1

#if 0
#define HSAIL_DEBUG(code) code;
#else
#define HSAIL_DEBUG(code)
#endif

class HSAILCodeGenContext : public WrappingCodeGenContext {
public:
    Triple TheTriple;

    HSAILCodeGenContext(CodeGenContext *inner)
        : WrappingCodeGenContext(inner) {}
};

#define hsail_ctx() ((HSAILCodeGenContext *)targetCodeGenContexts[HSAIL])

static std::unique_ptr<Module> load_hsail_intrinsics() {
    static std::unique_ptr<Module> MIntrin;

    if (!MIntrin) {
        // load intrinsics file the first time we are called
        HSAIL_DEBUG(errs() << "HSAIL: Loading HSAIL Intrinsics"
                           << "\n");

        const auto SearchPath = std::vector<const char *>{
            "/opt/amd/bin", getenv("HSA_BUILTINS_PATH"),
            getenv("HSA_RUNTIME_PATH")};
        const auto FileName = "/builtins-hsail.sign.bc";

        std::error_code ec;
        std::unique_ptr<MemoryBuffer> MB;

        bool isDir = false;
        for (const auto D : SearchPath) {
            if (D == nullptr)
                continue;

            ec = sys::fs::is_directory(D, isDir);
            if (!ec && isDir) {
                const auto FilePath = std::string(D) + FileName;
                HSAIL_DEBUG(errs() << "HSAIL: Searching path " << FilePath
                                   << "\n");

                auto MBoE = MemoryBuffer::getFile(FilePath);
                if (MBoE) {
                    HSAIL_DEBUG(errs() << "HSAIL: Found at " << FilePath
                                       << "\n");
                    MB = std::move(*MBoE);
                    break;
                } else {
                    HSAIL_DEBUG(errs() << "HSAIL: Not Found at " << FilePath
                                       << "\n");
                    ec = MBoE.getError();
                }
            }
        }

        if (!MB) {
            jl_errorf("Cannot open HSAIL builtins bitcode file: %s",
                      ec.message().c_str());
        }
        auto M2OrError =
            parseBitcodeFile(MB->getMemBufferRef(), getGlobalContext());

        if (std::error_code ec = M2OrError.getError()) {
            jl_errorf("could not parse device library: %s",
                      ec.message().c_str());
        }

        MIntrin = std::move(*M2OrError);

        auto CTX = hsail_ctx();
        MIntrin->setDataLayout(
            CTX->TM->getDataLayout()->getStringRepresentation());
    }

    return std::unique_ptr<Module>(llvm::CloneModule(MIntrin.get()));
}

extern "C" DLLEXPORT void jl_init_hsail_codegen(void) {
    if (hsail_ctx() != nullptr) {
        jl_error("cannot re-initialize HSAIL codegen, destroy the existing one "
                 "first");
    }

    if (!spir_ctx()) {
        jl_init_spir_codegen();
    }

    auto CTX = std::make_unique<HSAILCodeGenContext>(spir_ctx());

    Triple TheTriple("hsail64-unknown-unknown");

    auto TM = GetTargetMachine(TheTriple);

    if (!TM)
        jl_error("Could not create HSAIL target machine");

    CTX->TM = TM;

    targetCodeGenContexts[HSAIL] = CTX.release();

    HSAIL_DEBUG(jl_printf(JL_STDERR, "HSAIL codegen initialized\n"));
}

extern "C" DLLEXPORT Function *to_hsail(jl_lambda_info_t *li) {
    auto CTX = hsail_ctx();
    if (!CTX) {
        jl_error("HSAIL Codegen not initialized");
    }
    auto FSpir = (Function *)li->targetFunctionObjects[SPIR];

    if (!FSpir)
        FSpir = to_spir(li);

    auto MSpir = FSpir->getParent();
    // FIXME: We lose metadata nodes and more when copying to a new module
    //        For now, use the SPIR Module which prevents us from having SPIR
    //        and HSA IR at the same time
    // auto MHsail = std::unique_ptr<llvm::Module>(llvm::CloneModule(MSpir));
    auto MHsail = std::unique_ptr<llvm::Module>(MSpir);

    auto MIntrin = load_hsail_intrinsics();
    MHsail->setTargetTriple(MIntrin->getTargetTriple());
    MHsail->setDataLayout(MIntrin->getDataLayout());

#ifdef LLVM36
    // TODO: use the DiagnosticHandler
    if (Linker::LinkModules(MHsail.get(), MIntrin.release()))
        jl_error("Could not link device library");
#else
    jl_error("unsupported LLVM");
#endif

    // Run module passes again
    // (mainly to get inlining)
    auto PM = CTX->getModulePasses(MHsail.get());
    PM->run(*MHsail);

    auto FHsail = MHsail->getFunction(FSpir->getName());
    li->targetFunctionObjects[HSAIL] = FHsail;

    DEBUG_IF(HSA_IR, MHsail->dump());

    MHsail.release();

    return FHsail;
}

extern "C" DLLEXPORT void *jl_get_hsailf(jl_function_t *f, jl_tupletype_t *tt) {
    jl_function_t *sf = get_function_spec(f, tt);
    if (sf->linfo->targetFunctionObjects[HSAIL] == nullptr) {
        to_hsail(sf->linfo);
    }

    return (Function *)sf->linfo->targetFunctionObjects[HSAIL];
}

SmallVector<char, 4096> *to_brig(void *f) {
    auto CTX = hsail_ctx();
    auto FHsail = (Function *)f;
    auto MHsail = FHsail->getParent();

    auto PM = std::make_unique<PassManager>();

    // Write the assembly
    auto ObjBufferSV = std::make_unique<SmallVector<char, 4096>>();
    raw_svector_ostream OS(*ObjBufferSV);

    llvm::verifyModule(*MHsail, &errs());

    CTX->TM->addPassesToEmitFile(*PM, OS, TargetMachine::CGFT_ObjectFile, false,
                                 nullptr, nullptr);
    try {
        PM->run(*MHsail);
    } catch (const SyntaxError &err) {
        HSAIL_DEBUG(errs() << "HSAIL: " << err.what() << "\n");
    }
    OS.flush();

    return ObjBufferSV.release();
}

extern "C" DLLEXPORT void *jl_get_brigf(jl_function_t *f, jl_tupletype_t *tt) {
    jl_function_t *sf = get_function_spec(f, tt);
    if (sf->linfo->targetFunctionObjects[BRIG] == nullptr) {
        auto FHsail = jl_get_hsailf(f, tt);
        sf->linfo->targetFunctionObjects[BRIG] = to_brig(FHsail);
    }

    return ((SmallVector<char, 4096> *)sf->linfo->targetFunctionObjects[BRIG])
        ->data();
}

extern "C" DLLEXPORT jl_value_t *jl_dump_function_hsail(void *f) {
    auto CTX = hsail_ctx();
    auto FHsail = (Function *)f;
    auto MHsail = FHsail->getParent();

    auto PM = std::make_unique<PassManager>();

    // Write the assembly
    SmallVector<char, 4096> ObjBufferSV;
    raw_svector_ostream OS(ObjBufferSV);

    llvm::verifyModule(*MHsail, &errs());

    CTX->TM->addPassesToEmitFile(*PM, OS, TargetMachine::CGFT_AssemblyFile,
                                 false, nullptr, nullptr);
    try {
        PM->run(*MHsail);
    } catch (const SyntaxError &err) {
        HSAIL_DEBUG(errs() << "HSAIL: " << err.what() << "\n");
    }
    OS.flush();

    return jl_cstr_to_string(ObjBufferSV.data());
}
#endif
