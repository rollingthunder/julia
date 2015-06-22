class SPIRCodeGenContext : public CodeGenContext
{
public:
	std::map<jl_lambda_info_t*, Module*> Modules;

	Module* getModuleFor(jl_lambda_info_t* li) override {
		auto M = new Module(li->name->name, getGlobalContext());

		Modules[li] = M;

		return M;
	}

	void addMetadata(Function* f, jl_codectx_t& ctx) override {
        if (strncmp(ctx.linfo->name->name, "kernel_", 7) == 0) {
            Metadata *Elts[] = {
				ConstantAsMetadata::get(f),
				MDString::get(getGlobalContext(), "kernel"),
				ConstantAsMetadata::get(ConstantInt::get(T_size, 1))
			};
            MDNode *Node = MDNode::get(getGlobalContext(), Elts);

			auto m = f->getParent();
            NamedMDNode *NMD = m->getOrInsertNamedMetadata("nvvm.annotations");
            NMD->addOperand(Node);
        }
	}

	~SPIRCodeGenContext() override {
		for(const auto& KV : Modules) {
			delete KV.second;
		}
	}
};

#define spir_ctx() ((SPIRCodeGenContext*) targetCodeGenContexts[SPIR])

extern "C" DLLEXPORT
void jl_init_spir_codegen(void)
{
    if (spir_ctx() != nullptr)
	{
        jl_error("cannot re-initialize SPIR codegen, destroy the existing one first");
	}
	auto ctx = std::unique_ptr<SPIRCodeGenContext>( new SPIRCodeGenContext());

	std::string TheTriple("spir64-unknown-unknown");

//	std::string Error;
//	auto TheTarget = TargetRegistry::lookupTarget(TheTriple,
//			Error);
//	if (!TheTarget) {
//		jl_error("couldn't retrieve SPIR target");
//	}
//    auto TM = TheTarget->createTargetMachine(
//        TheTriple, /*CPU*/ "", "", TargetOptions(),
//        Reloc::PIC_, CodeModel::Default, CodeGenOpt::Aggressive);
//
//    if (!TM)
//        jl_error("Could not create SPIR target machine");
//    TM->setAsmVerbosityDefault(true);

    auto PM = std::unique_ptr<PassManager>( new PassManager());

    // Add the target data from the target machine
    PM->add(new DataLayoutPass());

    // Eliminate all unused functions
    PM->add(createGlobalOptimizerPass());
    PM->add(createStripDeadPrototypesPass());

	// Inline all functions with always_inline attribute
    PM->add(createAlwaysInlinerPass());

    auto FPM = std::unique_ptr<FunctionPassManager>( new FunctionPassManager(nullptr /*M*/));

    // Enqueue standard optimizations
    PassManagerBuilder PMB;
    PMB.OptLevel = CodeGenOpt::Aggressive;
    PMB.populateFunctionPassManager(*FPM);

//	ctx->TM = std::move(TM);
	ctx->FPM = std::move(FPM);
	ctx->PM = std::move(PM);

	targetCodeGenContexts[SPIR] = ctx.release();
}

static Module* load_hsail_intrinsics()
{
    const auto SearchPath = std::vector<const char*>{
		"/opt/amd/bin",
		getenv("HSA_BUILTINS_PATH"),
		getenv("HSA_RUNTIME_PATH")
	};
	const auto FileName = "builtins-hsail.sign.bc";

	std::error_code ec;
    std::unique_ptr<MemoryBuffer> MB;

    bool isDir = false;
    for (const auto D : SearchPath) {
        if (D == NULL)
            continue;

        ec = sys::fs::is_directory(D, isDir);
        if (!ec && isDir) {
			const auto FilePath = std::string(D) + FileName;
			auto MBoE = MemoryBuffer::getFile(FilePath);
			if((ec = MBoE.getError()))
			{
				MB = std::move(*MBoE);
				break;
			}
        }
    }

    if (ec)
	{
        jl_errorf("Cannot open HSAIL builtins bitcode file: %s",
                  ec.message().c_str());
	}
    auto M2OrError = parseBitcodeFile(MB->getMemBufferRef(),
                                                  getGlobalContext());

    if (std::error_code ec = M2OrError.getError())
	{
        jl_errorf("could not parse device library: %s", ec.message().c_str());
	}

	return *M2OrError;
}

static Function *to_spir(jl_lambda_info_t *li)
{
	jl_printf(JL_STDERR, "Generating SPIR\n");

	auto CTX = spir_ctx();

	if(!CTX)
	{
		jl_errorf("SPIR code generator not initialized\n");
	}

	auto F = (Function*)li->specFunctionObject;

	/*
	auto M = F->getParent();

	auto PM = std::unique_ptr<PassManager>( new PassManager());
#if defined(LLVM36)
	PM->add(new DataLayoutPass());
#else
	jl_errorf("SPIR Target not supported on your version of LLVM");
#endif
	PM->add(createInternalizePass({ F->getName().data() }));
	PM->run(*M);

    // Run common passes
    for (auto F = M->begin(), E = M->end(); F != E; ++F) {
        CTX->FPM->run(*F);
	}
    CTX->PM->run(*M);
	*/

	li->targetFunctionObjects[SPIR] = F;

	return F;
}

extern "C" DLLEXPORT
void *jl_get_spirf(jl_function_t *f, jl_tupletype_t *tt)
{
    jl_function_t *sf = f;
    if (tt != NULL) {
        if (!jl_is_function(f) || !jl_is_gf(f)) {
            return NULL;
        }
        sf = jl_get_specialization(f, tt);
    }
    if (sf == NULL || sf->linfo == NULL) {
        sf = jl_method_lookup_by_type(jl_gf_mtable(f), tt, 0, 0);
        if (sf == jl_bottom_func) {
            return NULL;
        }
        jl_printf(JL_STDERR,
                  "Warning: Returned code may not match what actually runs.\n");
    }
    if (sf->linfo->targetFunctionObjects[SPIR] == NULL) {
        to_spir(sf->linfo);
    }

	return (Function*)sf->linfo->targetFunctionObjects[SPIR];
}

#if HAS_HSA

#define hsail_ctx() ((HSAILCodeGenContext*) targetCodeGenContexts[HSAIL])

static void* to_brig(jl_lambda_info_t* li)
{
    M2->setTargetTriple(M->getTargetTriple());
#ifdef LLVM36
    // TODO: use the DiagnosticHandler
    if (Linker::LinkModules(M, M2))
        jl_error("Could not link device library");
#else
	jl_error("unsupported LLVM");
#endif
    delete M2;


    // Write the assembly
    std::string code;
    llvm::raw_string_ostream stream(code);
    llvm::formatted_raw_ostream fstream(stream);
    CTX->TM->addPassesToEmitFile(
        *PM, fstream, TargetMachine::CGFT_AssemblyFile, true, 0, 0);
    PM->run(*M);
    fstream.flush();
    stream.flush();

	// TODO: return BRIG
	return nullptr;
}
#endif
