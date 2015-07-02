namespace LangAS
{
// See clang/lib/Basic/Targets.cpp
// We are directly using the Target AddressSpaces
// and not the logical ones because we deal directly with
// the Target IR (SPIR)
enum SPIRAddressSpaces
{
	opencl_global = 1,
    opencl_local = 3,
    opencl_constant = 2,
    opencl_generic = 4
};

}

static TargetMachine* GetTargetMachine(Triple TheTriple) {
	std::string Error;
	auto MArch = TheTriple.getArchName();

	auto TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
			Error);
	// Some modules don't specify a triple, and this is okay.
	if (!TheTarget) {
		errs() << "Target Not Found \n";
		return nullptr;
	}

	// Package up features to be passed to target/subtarget
	std::string FeaturesStr;
	std::string MCPU;


	return TheTarget->createTargetMachine(TheTriple.getTriple(),
			MCPU, FeaturesStr,
			TargetOptions(),
			Reloc::Default, CodeModel::Default,
			CodeGenOpt::Aggressive);
}

class SPIRCodeGenContext : public CodeGenContext
{
private:
	std::map<Type*, std::string> MetadataTypeMap;

	void genOpenCLArgMetadata(llvm::Function *Fn,
		SmallVector<llvm::Metadata *, 5> &kernelMDArgs);
	void emitOpenCLKernelMetadata(llvm::Function *Fn);
	bool isKernel(Function& F) { return F.getReturnType() == T_void; }
	void addModuleMetadata(Module* M);
public:
	std::map<jl_lambda_info_t*, Module*> Modules;

	SPIRCodeGenContext();

	void runOnModule(Module* M) override;
	void updateFunctionSignature(std::vector<Type*>& argTypes, Type*& retType);
	virtual std::unique_ptr<PassManager> getModulePasses(Module* M) override;
	virtual std::unique_ptr<FunctionPassManager> getFunctionPasses(Module* M) override;

	Module* getModuleFor(jl_lambda_info_t* li) override {
		auto M = new Module(li->name->name, getGlobalContext());

		Modules[li] = M;

		return M;
	}

	void addMetadata(Function* f, jl_codectx_t& ctx) override;


	~SPIRCodeGenContext() override {
		for(const auto& KV : Modules) {
			delete KV.second;
		}
	}
};

SPIRCodeGenContext::SPIRCodeGenContext()
{
	std::string TheTriple("spir64-unknown-unknown");

	// Add a few supported types for metadata generation
	MetadataTypeMap = {
		{ T_char, "char"},
		{ T_uint8, "uchar"},
		{ T_int16, "short"},
		{ T_uint16, "ushort"},
		{ T_int32, "int" },
		{ T_uint32, "uint" },
		{ T_int64, "long"},
		{ T_uint64, "ulong"},
		{ T_float32, "float"},
		{ T_float64, "double"}
	};
}

void SPIRCodeGenContext::addMetadata(Function* F, jl_codectx_t& ctx) {
	// Add "spir_kernel" attribute
	F->addFnAttr("spir_kernel");


	emitOpenCLKernelMetadata(F);
}

void SPIRCodeGenContext::updateFunctionSignature(std::vector<Type*>& argTypes, Type*& retType)
{
	if (!retType->isVoidTy())
		jl_error("Only kernel functions returning void are supported");

	// for now, only global ptr args
	// update types to add the addr space
	for(auto& T : argTypes)
	{
		if (T->isPointerTy())
		{
			auto TAddr = PointerType::get(T->getPointerElementType(),
					LangAS::opencl_global);
			T = TAddr;
		}
	}
}

void SPIRCodeGenContext::runOnModule(Module* M)
{

}

std::string printTypeToString(Type* Ty)
{
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
void SPIRCodeGenContext::genOpenCLArgMetadata(llvm::Function *Fn,
		SmallVector<llvm::Metadata *, 5> &kernelMDArgs) {
	auto& Context = getGlobalContext();
  // Create MDNodes that represent the kernel arg metadata.
  // Each MDNode is a list in the form of "key", N number of values which is
  // the same number of values as their are kernel arguments.

  // MDNode for the kernel argument address space qualifiers.
  SmallVector<llvm::Metadata *, 8> addressQuals;
  addressQuals.push_back(llvm::MDString::get(Context, "kernel_arg_addr_space"));

  // MDNode for the kernel argument access qualifiers (images only).
  SmallVector<llvm::Metadata *, 8> accessQuals;
  accessQuals.push_back(llvm::MDString::get(Context, "kernel_arg_access_qual"));

  // MDNode for the kernel argument type names.
  SmallVector<llvm::Metadata *, 8> argTypeNames;
  argTypeNames.push_back(llvm::MDString::get(Context, "kernel_arg_type"));

  // MDNode for the kernel argument base type names.
  SmallVector<llvm::Metadata *, 8> argBaseTypeNames;
  argBaseTypeNames.push_back(
      llvm::MDString::get(Context, "kernel_arg_base_type"));

  // MDNode for the kernel argument type qualifiers.
  SmallVector<llvm::Metadata *, 8> argTypeQuals;
  argTypeQuals.push_back(llvm::MDString::get(Context, "kernel_arg_type_qual"));

  // MDNode for the kernel argument names.
  SmallVector<llvm::Metadata *, 8> argNames;
  argNames.push_back(llvm::MDString::get(Context, "kernel_arg_name"));

  for (auto& parm : Fn->args()) {
    auto ty = parm.getType();
    std::string typeQuals;

    std::string typeName("");
    if (ty->isPointerTy()) {
      auto pointeeTy = ty->getPointerElementType();

      // Get address qualifier.
      addressQuals
		  .push_back(
				  llvm::ConstantAsMetadata::get(
					  ConstantInt::get(T_int32,
						  cast<PointerType>(pointeeTy)->getAddressSpace()
						  )
					  )
				  );

      // Get argument type name.
	  auto name = MetadataTypeMap.find(pointeeTy);
	  if (name != MetadataTypeMap.end())
	  {
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

      addressQuals.push_back(
          llvm::ConstantAsMetadata::get(
			  ConstantInt::get(T_int32, AddrSpc)));

      // Get argument type name.
	  auto name = MetadataTypeMap.find(ty);
	  if (name != MetadataTypeMap.end())
	  {
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
	if (typeName == "")
	{
		typeName = (ty->isPointerTy()) ? "void*" : "int";
		errs() << "Argument type '"
			<< printTypeToString(ty)
			<< "' not found, emitting "
			<< typeName << " metadata";
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

NamedMDNode* getOrInsertOpenCLKernelNode(Module* M)
{
    return M->getOrInsertNamedMetadata("opencl.kernels");
}

Function* getKernelFromMDNode(NamedMDNode* KernelMD, size_t i)
{
	auto NKernels = KernelMD->getNumOperands();

	if(i < NKernels) {
		auto KernelNode = cast_or_null<ValueAsMetadata>(KernelMD->getOperand(i));
		if(KernelNode)
		{
            auto Kernel = cast_or_null<Function>(KernelNode->getValue());
			if(Kernel)
			{
				return Kernel;
			}
		}
	}

	return nullptr;
}

void SPIRCodeGenContext::emitOpenCLKernelMetadata(llvm::Function *Fn)
{
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

std::unique_ptr<PassManager> SPIRCodeGenContext::getModulePasses(Module* M)
{
	auto PM = CodeGenContext::getModulePasses(M);

	SmallVector<std::string, 8> NameStrings;
	SmallVector<const char*, 8> ExportedNames;

	auto I = 0U;
	for(auto& F : *M)
	{
		NameStrings.push_back(F.getName());
		ExportedNames.push_back(NameStrings[I].c_str());
		I++;
	}

	PM->add(createInternalizePass(ArrayRef<const char*>(ExportedNames)));

	return PM;
}

#define spir_ctx() ((SPIRCodeGenContext*) targetCodeGenContexts[SPIR])

extern "C" DLLEXPORT
void jl_init_spir_codegen(void)
{
    if (spir_ctx() != nullptr)
	{
        jl_error("cannot re-initialize SPIR codegen, destroy the existing one first");
	}

	targetCodeGenContexts[SPIR] = new SPIRCodeGenContext();
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

	auto M = F->getParent();

	CTX->runOnModule(M);

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

static std::unique_ptr<Module> load_hsail_intrinsics()
{
	static std::unique_ptr<Module> MIntrin;

	if (!MIntrin) {
		// load intrinsics file the firs time we are called

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

		MIntrin = *M2OrError;
	}

	return std::unique_ptr<Module>(llvm::CloneModule(MIntrin.get()));
}

class HSAILCodeGenContext : public WrappingCodeGenContext {
public:
	Triple TheTriple;

	HSAILCodeGenContext(CodeGenContext* inner) : WrappingCodeGenContext(inner) {}
};

#define hsail_ctx() ((HSAILCodeGenContext*) targetCodeGenContexts[HSAIL])

struct DisposeLogger {
	std::string name;
	bool released;

	DisposeLogger(std::string name) : name(name), released(false) {}

	~DisposeLogger()
	{
		if(!released)
			jl_printf(JL_STDERR, "Disposed: %s", name.c_str());
	}
};


extern "C" DLLEXPORT
void jl_init_hsail_codegen(void)
{
    if (hsail_ctx() != nullptr)
	{
        jl_error("cannot re-initialize HSAIL codegen, destroy the existing one first");
	}

	if(!spir_ctx()){
		jl_init_spir_codegen();
	}

	auto CTX = std::make_unique<HSAILCodeGenContext>(spir_ctx());

	Triple TheTriple("hsail64-unknown-unknown");

	auto TM = GetTargetMachine(TheTriple);

	if (!TM)
		jl_error("Could not create HSAIL target machine");

	targetCodeGenContexts[HSAIL] = CTX.release();
}

extern "C" DLLEXPORT
Function* to_hsail(jl_lambda_info_t* li)
{
    auto FSpir = to_spir(li);
	auto MSpir = FSpir->getParent();
	auto MHsail = std::unique_ptr<llvm::Module>(llvm::CloneModule(MSpir));
	auto logger = DisposeLogger("before_intrinsics");
	auto MIntrin = load_hsail_intrinsics();
	MHsail->setTargetTriple(MIntrin->getTargetTriple());

#ifdef LLVM36
    // TODO: use the DiagnosticHandler
    if (Linker::LinkModules(MHsail.get(), MIntrin.release()))
        jl_error("Could not link device library");
#else
	jl_error("unsupported LLVM");
#endif

	logger.released = true;

	auto FHsail = MHsail->getFunction(FSpir->getName());
	li->targetFunctionObjects[HSAIL] = FHsail;

	MHsail.release();

	return FHsail;
}

extern "C" DLLEXPORT
void *jl_get_hsailf(jl_function_t *f, jl_tupletype_t *tt)
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
    if (sf->linfo->targetFunctionObjects[HSAIL] == NULL) {
        to_spir(sf->linfo);
    }

	return (Function*)sf->linfo->targetFunctionObjects[SPIR];
}

extern "C" DLLEXPORT
jl_value_t* jl_dump_function_hsail(void* f)
{
	auto CTX = hsail_ctx();
	auto FHsail = (Function*)f;
	auto MHsail = FHsail->getParent();

	auto PM = std::make_unique<PassManager>();

    // Write the assembly
	SmallVector<char, 4096> ObjBufferSV;
	raw_svector_ostream OS(ObjBufferSV);

    CTX->TM->addPassesToEmitFile(
        *PM, OS, TargetMachine::CGFT_AssemblyFile, true, 0, 0);
    PM->run(*MHsail);
    OS.flush();

    return jl_cstr_to_string(ObjBufferSV.data());
}
#endif
