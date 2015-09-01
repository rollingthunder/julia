#define DEBUG_IF(COND, CODE)                                                   \
    if (COND) {                                                                \
        CODE;                                                                  \
    }

class CodeGenContext {
public:
    // commmon components for a device target codegen
    TargetMachine *TM;

    virtual void updateFunctionSignature(jl_lambda_info_t *li,
                                         std::stringstream &fName,
                                         std::vector<Type *> &argTypes,
                                         Type *&retType) {}
    virtual std::unique_ptr<PassManager> getModulePasses(Module *M);
    virtual std::unique_ptr<FunctionPassManager> getFunctionPasses(Module *M);
    virtual void runOnModule(Module *M);
    virtual Module *getModuleFor(jl_lambda_info_t *li) = 0;
    virtual Function *generateWrapper(jl_lambda_info_t *li, jl_expr_t *ast,
                                      Function *f);
    virtual void addMetadata(Function *f, jl_codectx_t &ctx) {}
    virtual ~CodeGenContext() {}
};

void CodeGenContext::runOnModule(Module *M) {
    // run module passes, if any
    auto PM = getModulePasses(M);
    if (PM) {
        PM->run(*M);
    }

    // run function passes
    auto FPM = getFunctionPasses(M);
    if (FPM) {
        FPM->doInitialization();
        for (auto &F : *M) {
            FPM->run(F);
        }
        FPM->doFinalization();
    }
}

std::unique_ptr<PassManager> CodeGenContext::getModulePasses(Module *M) {
    auto PM = std::make_unique<PassManager>();

    // Eliminate all unused functions
    PM->add(createGlobalOptimizerPass());
    PM->add(createStripDeadPrototypesPass());

    // Inline all functions with always_inline attribute
    PM->add(createAlwaysInlinerPass());

    return PM;
}

std::unique_ptr<FunctionPassManager>
CodeGenContext::getFunctionPasses(Module *M) {
    auto FPM = std::make_unique<FunctionPassManager>(M);

    // Enqueue standard optimizations
    PassManagerBuilder PMB;
    PMB.OptLevel = CodeGenOpt::Aggressive;
    PMB.populateFunctionPassManager(*FPM);

    return FPM;
}
class WrappingCodeGenContext : public CodeGenContext {
private:
    CodeGenContext *Inner;

public:
    WrappingCodeGenContext(CodeGenContext *inner) : Inner(inner) {}
    virtual void updateFunctionSignature(jl_lambda_info_t *li,
                                         std::stringstream &fName,
                                         std::vector<Type *> &argTypes,
                                         Type *&retType) override {
        Inner->updateFunctionSignature(li, fName, argTypes, retType);
    }
    virtual void runOnModule(Module *M) override { Inner->runOnModule(M); }
    virtual Module *getModuleFor(jl_lambda_info_t *li) override {
        return Inner->getModuleFor(li);
    }
    virtual Function *generateWrapper(jl_lambda_info_t *li, jl_expr_t *ast,
                                      Function *f) override {
        return Inner->generateWrapper(li, ast, f);
    }
    virtual void addMetadata(Function *f, jl_codectx_t &ctx) {
        Inner->addMetadata(f, ctx);
    }
    virtual ~WrappingCodeGenContext() {}
};
