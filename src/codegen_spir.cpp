#if 1
#define SPIR_DEBUG(code) code;
#else
#define SPIR_DEBUG(code)
#endif

#define SP_IR 1

#include "codegen_spir_passes.cpp"

jl_function_t *get_function_spec(jl_function_t *f, jl_tupletype_t *tt) {
    jl_function_t *sf = f;
    if (tt != nullptr) {
        if (!jl_is_function(f) || !jl_is_gf(f)) {
            return nullptr;
        }
        sf = jl_get_specialization(f, tt);
    }
    if (sf == nullptr || sf->linfo == nullptr) {
        sf = jl_method_lookup_by_type(jl_gf_mtable(f), tt, 0, 0);
        if (sf == jl_bottom_func) {
            return nullptr;
        }
        jl_printf(JL_STDERR,
                  "Warning: Returned code may not match what actually runs.\n");
    }

    return sf;
}

static TargetMachine *GetTargetMachine(Triple TheTriple) {
    std::string Error;
    auto MArch = TheTriple.getArchName();

    auto TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple, Error);
    // Some modules don't specify a triple, and this is okay.
    if (!TheTarget) {
        errs() << "Target Not Found \n";
        return nullptr;
    }

    // Package up features to be passed to target/subtarget
    std::string FeaturesStr;
    std::string MCPU;

    return TheTarget->createTargetMachine(
        TheTriple.getTriple(), MCPU, FeaturesStr, TargetOptions(),
        Reloc::Default, CodeModel::Default, CodeGenOpt::Aggressive);
}

class SPIRCodeGenContext : public CodeGenContext {
private:
    std::string TheTriple;
    std::map<Type *, std::string> MetadataTypeMap;

    void genOpenCLArgMetadata(llvm::Function *Fn,
                              SmallVector<llvm::Metadata *, 5> &kernelMDArgs);
    void emitOpenCLKernelMetadata(llvm::Function *Fn);
    bool isKernel(Function *F) { return F->getReturnType()->isVoidTy(); }
    void addModuleMetadata(Module *M);

public:
    std::map<jl_lambda_info_t *, Module *> Modules;

    SPIRCodeGenContext();

    void runOnModule(Module *M) override;
    void updateFunctionSignature(jl_lambda_info_t *li, std::stringstream &fName,
                                 std::vector<Type *> &argTypes, Type *&retType);
    virtual std::unique_ptr<PassManager> getModulePasses(Module *M) override;

    Module *getModuleFor(jl_lambda_info_t *li) override {
        auto M = new Module(li->name->name, getGlobalContext());

        M->addModuleFlag(llvm::Module::Warning, "Dwarf Version",2);
        M->addModuleFlag(llvm::Module::Error, "Debug Info Version",
            llvm::DEBUG_METADATA_VERSION);
        M->setTargetTriple(TheTriple);

        Modules[li] = M;

        return M;
    }

    void addMetadata(Function *f, jl_codectx_t &ctx) override;

    ~SPIRCodeGenContext() override {
        for (const auto &KV : Modules) {
            delete KV.second;
        }
    }
};

SPIRCodeGenContext::SPIRCodeGenContext() : TheTriple("spir64-unknown-unknown") {
    // Add a few supported types for metadata generation
    MetadataTypeMap = {{T_char, "char"},
                       {T_uint8, "uchar"},
                       {T_int16, "short"},
                       {T_uint16, "ushort"},
                       {T_int32, "int"},
                       {T_uint32, "uint"},
                       {T_int64, "long"},
                       {T_uint64, "ulong"},
                       {T_float32, "float"},
                       {T_float64, "double"}};
}

void SPIRCodeGenContext::addMetadata(Function *F, jl_codectx_t &ctx) {
    SPIR_DEBUG(errs() << "SPIR: Adding Metadata for " << F->getName() << "\n")

    F->setCallingConv(CallingConv::SPIR_KERNEL);
        //F->setCallingConv(CallingConv::SPIR_FUNC);

    emitOpenCLKernelMetadata(F);
}

void SPIRCodeGenContext::updateFunctionSignature(jl_lambda_info_t *li,
                                                 std::stringstream &fName,
                                                 std::vector<Type *> &argTypes,
                                                 Type *&retType) {
    SPIR_DEBUG(errs() << "SPIR: Updating function signature\n");

    // Strip pre- & suffixes from function name
    fName.str("");
    fName.clear();
    fName << li->name->name;

    if (!retType->isVoidTy())
        jl_error("Only kernel functions returning void are supported");

    // for now, only global ptr args
    // update types to add the addr space
    for (size_t I = 0U, E = argTypes.size(); I < E; ++I) {
        auto T = argTypes[I];
        if (T->isPointerTy()) {
            auto TAddr = PointerType::get(T->getPointerElementType(),
                                          LangAS::opencl_global);
            // TODO: Fix julia issues when using argument types with
            // AddressSpace
            // argTypes[I] = TAddr;
        }
    }
}

void SPIRCodeGenContext::runOnModule(Module *M) {
    CodeGenContext::runOnModule(M);
}

std::string printTypeToString(Type *Ty) {
    std::string buf;
    raw_string_ostream os(buf);

    if (Ty)
        Ty->print(os);
    else
        os << "Printing <null> Type";

    os.flush();

    return buf;
}

// Adapted Metadata Emission from Khronos/SPIR project
// OpenCL v1.2 s5.6.4.6 allows the compiler to store kernel argument
// information in the program executable. The argument information stored
// includes the argument name, its type, the address and access qualifiers used.
void SPIRCodeGenContext::genOpenCLArgMetadata(
    llvm::Function *Fn, SmallVector<llvm::Metadata *, 5> &kernelMDArgs) {
    auto &Context = getGlobalContext();
    // Create MDNodes that represent the kernel arg metadata.
    // Each MDNode is a list in the form of "key", N number of values which is
    // the same number of values as their are kernel arguments.

    // MDNode for the kernel argument address space qualifiers.
    SmallVector<llvm::Metadata *, 8> addressQuals;
    addressQuals.push_back(
        llvm::MDString::get(Context, "kernel_arg_addr_space"));

    // MDNode for the kernel argument access qualifiers (images only).
    SmallVector<llvm::Metadata *, 8> accessQuals;
    accessQuals.push_back(
        llvm::MDString::get(Context, "kernel_arg_access_qual"));

    // MDNode for the kernel argument type names.
    SmallVector<llvm::Metadata *, 8> argTypeNames;
    argTypeNames.push_back(llvm::MDString::get(Context, "kernel_arg_type"));

    // MDNode for the kernel argument base type names.
    SmallVector<llvm::Metadata *, 8> argBaseTypeNames;
    argBaseTypeNames.push_back(
        llvm::MDString::get(Context, "kernel_arg_base_type"));

    // MDNode for the kernel argument type qualifiers.
    SmallVector<llvm::Metadata *, 8> argTypeQuals;
    argTypeQuals.push_back(
        llvm::MDString::get(Context, "kernel_arg_type_qual"));

    // MDNode for the kernel argument names.
    SmallVector<llvm::Metadata *, 8> argNames;
    argNames.push_back(llvm::MDString::get(Context, "kernel_arg_name"));

    for (auto &parm : Fn->args()) {
        auto ty = parm.getType();
        std::string typeQuals;

        std::string typeName("");
        if (ty->isPointerTy()) {
            auto pointeeTy = ty->getPointerElementType();

            // Get address qualifier.
            addressQuals.push_back(
                llvm::ConstantAsMetadata::get(ConstantInt::get(
                    T_int32, cast<PointerType>(ty)->getAddressSpace())));

            // Get argument type name.
            auto name = MetadataTypeMap.find(pointeeTy);
            if (name != MetadataTypeMap.end()) {
                typeName = name->second + "*";
            }

            /*/ Get argument type qualifiers:
            if (ty.isRestrictQualified())
              typeQuals = "restrict";
            if (pointeeTy.isConstQualified() ||
                (pointeeTy.getAddressSpace() == LangAS::opencl_constant))
              typeQuals += typeQuals.empty() ? "const" : " const";
            if (pointeeTy.isVolatileQualified())
              typeQuals += typeQuals.empty() ? "volatile" : " volatile";
              */
        } else {
            uint32_t AddrSpc = 0;
            /*
            if (ty->isImageType())
              AddrSpc = LangAS::opencl_global;
              */

            addressQuals.push_back(llvm::ConstantAsMetadata::get(
                ConstantInt::get(T_int32, AddrSpc)));

            // Get argument type name.
            auto name = MetadataTypeMap.find(ty);
            if (name != MetadataTypeMap.end()) {
                typeName = name->second;
            }

            /*/ Get argument type qualifiers:
            if (ty.isConstQualified())
              typeQuals = "const";
            if (ty.isVolatileQualified())
              typeQuals += typeQuals.empty() ? "volatile" : " volatile";
              */
        }
        argTypeQuals.push_back(llvm::MDString::get(Context, typeQuals));

        // Emit argument type
        if (typeName == "") {
            typeName = (ty->isPointerTy()) ? "void*" : "int";
            errs() << "Argument type '" << printTypeToString(ty)
                   << "' not found, emitting " << typeName << " metadata";
        }
        argTypeNames.push_back(llvm::MDString::get(Context, typeName));
        // TODO: actually emit the base type
        argBaseTypeNames.push_back(llvm::MDString::get(Context, typeName));

        /*/ Get image access qualifier:
    if (ty->isImageType()) {
      const OpenCLImageAccessAttr *A = parm->getAttr<OpenCLImageAccessAttr>();
      if (A && A->isWriteOnly())
        accessQuals.push_back(llvm::MDString::get(Context, "write_only"));
      else
        accessQuals.push_back(llvm::MDString::get(Context, "read_only"));
      // FIXME: what about read_write?
    } else*/ {
            accessQuals.push_back(llvm::MDString::get(Context, "none"));
        }

        // Get argument name.
        argNames.push_back(llvm::MDString::get(Context, parm.getName()));
    }

    kernelMDArgs.push_back(llvm::MDNode::get(Context, addressQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, accessQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argBaseTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argTypeQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argNames));
}

void SPIRCodeGenContext::emitOpenCLKernelMetadata(llvm::Function *Fn) {
    llvm::LLVMContext &Context = getGlobalContext();

    SmallVector<llvm::Metadata *, 5> kernelMDArgs;
    kernelMDArgs.push_back(llvm::ConstantAsMetadata::get(Fn));

    genOpenCLArgMetadata(Fn, kernelMDArgs);

    /*
    if (const VecTypeHintAttr *A = FD->getAttr<VecTypeHintAttr>()) {
      QualType hintQTy = A->getTypeHint();
      const ExtVectorType *hintEltQTy = hintQTy->getAs<ExtVectorType>();
      bool isSignedInteger =
          hintQTy->isSignedIntegerType() ||
          (hintEltQTy && hintEltQTy->getElementType()->isSignedIntegerType());
      llvm::Metadata *attrMDArgs[] = {
          llvm::MDString::get(Context, "vec_type_hint"),
          llvm::ConstantAsMetadata::get(llvm::UndefValue::get(
              CGM.getTypes().ConvertType(A->getTypeHint()))),
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              llvm::IntegerType::get(Context, 32),
              llvm::APInt(32, (uint64_t)(isSignedInteger ? 1 : 0))))};
      kernelMDArgs.push_back(llvm::MDNode::get(Context, attrMDArgs));
    }

    if (const WorkGroupSizeHintAttr *A = FD->getAttr<WorkGroupSizeHintAttr>()) {
      llvm::Metadata *attrMDArgs[] = {
          llvm::MDString::get(Context, "work_group_size_hint"),
          llvm::ConstantAsMetadata::get(Builder.getInt32(A->getXDim())),
          llvm::ConstantAsMetadata::get(Builder.getInt32(A->getYDim())),
          llvm::ConstantAsMetadata::get(Builder.getInt32(A->getZDim()))};
      kernelMDArgs.push_back(llvm::MDNode::get(Context, attrMDArgs));
    }

    if (const ReqdWorkGroupSizeAttr *A = FD->getAttr<ReqdWorkGroupSizeAttr>()) {
      llvm::Metadata *attrMDArgs[] = {
          llvm::MDString::get(Context, "reqd_work_group_size"),
          llvm::ConstantAsMetadata::get(Builder.getInt32(A->getXDim())),
          llvm::ConstantAsMetadata::get(Builder.getInt32(A->getYDim())),
          llvm::ConstantAsMetadata::get(Builder.getInt32(A->getZDim()))};
      kernelMDArgs.push_back(llvm::MDNode::get(Context, attrMDArgs));
    }
    */

    llvm::MDNode *kernelMDNode = llvm::MDNode::get(Context, kernelMDArgs);
    auto M = Fn->getParent();
    auto OpenCLKernelMetadata = getOrInsertOpenCLKernelNode(M);
    OpenCLKernelMetadata->addOperand(kernelMDNode);
}

std::unique_ptr<PassManager> SPIRCodeGenContext::getModulePasses(Module *M) {
    auto PM = CodeGenContext::getModulePasses(M);

    SmallVector<std::string, 8> NameStrings;
    SmallVector<const char *, 8> ExportedNames;

    auto OpenCLKernelMD = getOrInsertOpenCLKernelNode(M);
    auto N = OpenCLKernelMD->getNumOperands();
    for (auto I = 0U; I < N; ++I) {
        auto F = getKernelFromMDNode(OpenCLKernelMD, I);

        if (!F) {
            SPIR_DEBUG(errs()
                       << "SPIR: Could not retrieve Kernel from Metadata\n")
            continue;
        }

        SPIR_DEBUG(errs() << "SPIR: Kernel in Metadata - " << F->getName()
                          << "\n")

        NameStrings.push_back(F->getName());
        ExportedNames.push_back(NameStrings[I].c_str());
        I++;
    }

    //PM->add(new AddAddrSpacePass());
    PM->add(new SpirConvertSccPass());

    return PM;
}

#define spir_ctx() ((SPIRCodeGenContext *)targetCodeGenContexts[SPIR])

extern "C" DLLEXPORT void jl_init_spir_codegen(void) {
    if (spir_ctx() != nullptr) {
        jl_error("cannot re-initialize SPIR codegen, destroy the existing one "
                 "first");
    }

    targetCodeGenContexts[SPIR] = new SPIRCodeGenContext();

    SPIR_DEBUG(jl_printf(JL_STDERR, "SPIR codegen initialized\n"));
}

static Function *to_function(jl_lambda_info_t *li);

static Function *to_spir(jl_lambda_info_t *li) {
    SPIR_DEBUG(jl_printf(JL_STDERR, "SPIR: Generating\n"));

    auto CTX = spir_ctx();

    if (!CTX) {
        jl_errorf("SPIR code generator not initialized\n");
    }

    auto F = (Function *)li->specFunctionObject;

    if (!F) {
        F = to_function(li);
    }

    auto M = F->getParent();

    auto MSpir = llvm::CloneModule(M);

    // Running passes on the kernel may replace it
    // with another one, save thename so we can find it again later
    auto KernelName = F->getName();

    SPIR_DEBUG(errs() << "SPIR: Compiling " << KernelName << "\n");

    CTX->runOnModule(MSpir);

    F = MSpir->getFunction(KernelName);

    SPIR_DEBUG(errs() << "SPIR: IR Generation completed\n");

    li->targetFunctionObjects[SPIR] = F;

    return F;
}

extern "C" DLLEXPORT void *jl_get_spirf(jl_function_t *f, jl_tupletype_t *tt) {
    jl_function_t *sf = get_function_spec(f, tt);
    if (sf->linfo->targetFunctionObjects[SPIR] == nullptr) {
        to_spir(sf->linfo);
    }

    return (Function *)sf->linfo->targetFunctionObjects[SPIR];
}

extern "C" DLLEXPORT void jl_store_atomic_rel_u16(uint16_t *location,
                                                  uint16_t val) {
    __atomic_store_n(location, val, __ATOMIC_RELEASE);
}
