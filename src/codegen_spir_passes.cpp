namespace LangAS {
// See clang/lib/Basic/Targets.cpp
// We are directly using the Target AddressSpaces
// and not the logical ones because we deal directly with
// the Target IR (SPIR)
enum SPIRAddressSpaces {
    opencl_global = 1,
    opencl_local = 3,
    opencl_constant = 2,
    opencl_generic = 4
};
}

#include "llvm/Pass.h"
#include "llvm/IR/ValueMap.h"

struct AddAddrSpacePass : public ModulePass {
    static char PassID;

    AddAddrSpacePass() : ModulePass(PassID) {}

    bool runOnModule(Module &M) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesCFG();
    }

private:
    FunctionType *mapFunctionType(FunctionType *FTy);
    void replaceArgumentUses(Function *NewF, Function *F);
    void replaceValueWithAS(Value *NewV, Value *V);
};

char AddAddrSpacePass::PassID;

bool AddAddrSpacePass::runOnModule(Module &M) {
    bool Changed = false;
    for (auto F = M.begin(), FE = M.end(); F != FE; ++F) {

        auto FTy = F->getFunctionType();
        auto NewFTy = mapFunctionType(FTy);

        // Unchanged?
        if (FTy == NewFTy)
            continue;

        // Create replacement function
        auto NewF = Function::Create(NewFTy, F->getLinkage());
        NewF->copyAttributesFrom(F);
        M.getFunctionList().insert(F, NewF);
        NewF->takeName(F);

        // Take the Basic Blocks from the old Function
        // and move them to the new
        NewF->getBasicBlockList().splice(NewF->begin(), F->getBasicBlockList());

        replaceArgumentUses(NewF, F);
    }
    return Changed;
}

FunctionType *AddAddrSpacePass::mapFunctionType(FunctionType *FTy) {
    std::vector<Type *> PTypes;

    // Collect Parameters that need an address space
    for (auto P = FTy->param_begin(), PE = FTy->param_end(); P != PE; ++P) {
        auto PTy = *P;

        // Only pointer arguments can have an address space
        if (PTy->isPointerTy()) {
            // only arguments in the default address space
            // can possibly need conversion
            if (PTy->getPointerAddressSpace() == 0) {
                // Make all pointer arguments point to
                // the global address space
                // TODO Provide a mechanism to change that
                auto NewTy = PointerType::get(PTy->getPointerElementType(),
                                              LangAS::opencl_global);
                PTy = NewTy;
            }
        }

        PTypes.push_back(PTy);
    }

    if (FTy->isVarArg()) {
        jl_error("No VarArgs Kernels supported");
    }

    return FunctionType::get(FTy->getReturnType(), PTypes, FTy->isVarArg());
}

void AddAddrSpacePass::replaceArgumentUses(Function *NewF, Function *F) {
    for (auto A = F->arg_begin(), NA = NewF->arg_begin(), AE = F->arg_end();
         A != AE; A++, NA++) {
        replaceValueWithAS(NA, A);
    }
}

void AddAddrSpacePass::replaceValueWithAS(Value *NewV, Value *V) {
    auto Ty = V->getType();
    auto NewTy = NewV->getType();

    // Trivial case without conversion
    if (Ty == NewTy) {
        V->replaceAllUsesWith(NewV);
    } else if (Ty->isPointerTy() && NewTy->isPointerTy()) {
        for (auto U = V->use_begin(), UE = V->use_end(); U != UE; U++) {
            auto Usr = U->getUser();

            // Propagate the new address space across pointer-to-pointer casts
            if (auto Cast = dyn_cast<BitCastInst>(Usr)) {
                auto DestTy = Cast->getDestTy();
                if (DestTy->isPointerTy()) {
                    auto NewDestTy =
                        PointerType::get(DestTy->getPointerElementType(),
                                         NewTy->getPointerAddressSpace());
                    if (DestTy != NewDestTy) {
                        auto NewCast = new BitCastInst(NewV, NewDestTy, Cast->getName(), Cast);
                        replaceValueWithAS(NewCast, Cast);
                    } else {
                        U->set(NewV);
                    }
                }
            }
            // Propagate across gep instructions
            else if (auto GEP = dyn_cast<GetElementPtrInst>(Usr)) {
                // just changing the input value does not
                // update the type with the new address space

                SmallVector<Value *, 8> Indices(
                    make_range(GEP->idx_begin(), GEP->idx_end()));
                auto NewGEP =
                    GetElementPtrInst::Create(GEP->getResultElementType(), NewV,
                                              Indices, GEP->getName(), GEP);
                replaceValueWithAS(NewGEP, GEP);
                // GEP->eraseFromParent();
            }
            // for other uses, just replace the value
            else {
                U->set(NewV);
            }
        }
    } else {
        errs() << "AddAddrSpacePass: Unsupported Value replacement.";
    }
}

NamedMDNode *getOrInsertOpenCLKernelNode(Module *M) {
    return M->getOrInsertNamedMetadata("opencl.kernels");
}

Function *getKernelFromMDNode(NamedMDNode *KernelMD, size_t i) {
    auto NKernels = KernelMD->getNumOperands();

    if (i < NKernels) {
        auto KernelNode = KernelMD->getOperand(i);
        if (KernelNode) {
            auto FunctionValue =
                dyn_cast<ValueAsMetadata>(KernelNode->getOperand(0));
            if (FunctionValue) {
                auto Kernel = dyn_cast<Function>(FunctionValue->getValue());
                if (Kernel) {
                    return Kernel;
                }
            }
        }
    }

    return nullptr;
}

void setMDNodeKernelOperand(NamedMDNode *KernelMD, size_t i, Function *K) {
    auto NKernels = KernelMD->getNumOperands();

    if (i < NKernels) {
        auto KernelNode = KernelMD->getOperand(i);
        if (KernelNode) {
            SmallVector<llvm::Metadata *, 5> kernelMDArgs(
                KernelNode->operands());
            auto valueAsMD = ValueAsMetadata::get(K);
            kernelMDArgs[0] = valueAsMD;
            auto newKernelNode = MDNode::get(K->getContext(), kernelMDArgs);
            KernelMD->setOperand(i, newKernelNode);
        }
    }
}

class SpirConvertSccPass : public ModulePass {
    static char PassID;

    typedef std::vector<Value *> Values;

    Values deadInstructions;

    Value *mapValue(ValueToValueMapTy &VMap, ...);
    Function *mapFunction(ValueToValueMapTy &VMap, Module &M, ...);
    Function *mapKernel(ValueToValueMapTy &VMap, Module &M, Function *Kernel);

    FunctionType *mapKernelSignature(FunctionType *FTy);
    void propagateAddressSpace(Value* V);

public:
    SpirConvertSccPass() : ModulePass(PassID) {}

    bool runOnModule(Module &m) override;
};

char SpirConvertSccPass::PassID;

bool SpirConvertSccPass::runOnModule(Module &M) {
    ValueToValueMapTy VMap;

    auto oclKernels = getOrInsertOpenCLKernelNode(&M);

    for (auto i = 0U, num = oclKernels->getNumOperands(); i < num; ++i) {
        if (auto kernel = getKernelFromMDNode(oclKernels, i)) {
            auto newKernel = mapKernel(VMap, M, kernel);

            setMDNodeKernelOperand(oclKernels, i, newKernel);
            kernel->eraseFromParent();
        } else {
            errs() << "OpenCL Kernel Node Operand is not a Kernel\n";
        }
    }

    // We changed the Module
    return true;
}

Function *SpirConvertSccPass::mapKernel(ValueToValueMapTy &VMap, Module &M,
                                        Function *F) {
    auto NewFTy = mapKernelSignature(F->getFunctionType());

    // Create replacement function
    auto NewF = Function::Create(NewFTy, F->getLinkage(), "", &M);

    NewF->takeName(F);

    VMap[F] = NewF;

    // Add Mappings for the Arguments
    auto OldAI = F->arg_begin();
    for (auto AI = NewF->arg_begin(), AE = NewF->arg_end(); AI != AE;
         ++AI, ++OldAI) {
        VMap[OldAI] = AI;
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewF, F, VMap, /*ModuleLevelChanges =*/true, Returns, "",
                      nullptr);

    // The argument types' address spaces may have changed,
    // propagate the changes to make the function valid again
    for (auto AI = NewF->arg_begin(), AE = NewF->arg_end(); AI != AE;
         ++AI) {
        propagateAddressSpace(AI);
    }


    // TODO copy called functions to the kernel module
    // and update their signature if necessary

    return NewF;
}

FunctionType *SpirConvertSccPass::mapKernelSignature(FunctionType *FTy) {
    std::vector<Type *> PTypes;

    // Collect Parameters that need an address space
    for (auto P = FTy->param_begin(), PE = FTy->param_end(); P != PE; ++P) {
        auto PTy = *P;

        // Only pointer arguments can have an address space
        if (PTy->isPointerTy()) {
            // only arguments in the default address space
            // can possibly need conversion
            if (PTy->getPointerAddressSpace() == 0) {
                // Make all pointer arguments point to
                // the global address space
                // TODO Provide a mechanism to change that
                auto NewTy = PointerType::get(PTy->getPointerElementType(),
                                              LangAS::opencl_global);
                PTy = NewTy;
            }
        }

        PTypes.push_back(PTy);
    }

    if (FTy->isVarArg()) {
        jl_error("No VarArgs Kernels supported");
    }

    return FunctionType::get(FTy->getReturnType(), PTypes, FTy->isVarArg());
}

void SpirConvertSccPass::propagateAddressSpace(Value* V) {
    auto Ty = V->getType();

    if (Ty->isPointerTy()) {
        for (auto U = V->use_begin(), UE = V->use_end(); U != UE; U++) {
            auto Changed = false;
            auto Usr = U->getUser();
            auto UsrTy = Usr->getType();

            // Propagate the new address space across pointer-to-pointer casts
            if (auto Cast = dyn_cast<BitCastInst>(Usr)) {
                auto DestTy = Cast->getDestTy();
                if (DestTy->isPointerTy()) {
                    auto NewDestTy =
                        PointerType::get(DestTy->getPointerElementType(),
                                         Ty->getPointerAddressSpace());
                    if (DestTy != NewDestTy) {
                        Cast->mutateType(NewDestTy);
                        Changed = true;
                    }
                }
            }
            // Propagate across gep instructions
            else if (auto GEP = dyn_cast<GetElementPtrInst>(Usr)) {
                auto NewTy = PointerType::get(GEP->getResultElementType(),
                        Ty->getPointerAddressSpace());

                if(NewTy != UsrTy) {
                    GEP->mutateType(NewTy);
                    Changed = true;
                }
            }
            if(Changed) {
                propagateAddressSpace(Usr);
            }
        }
    }
}
