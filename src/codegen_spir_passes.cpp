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
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"

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

    Function *mapFunction(ValueToValueMapTy &VMap, Module &M, Function *F,
                          FunctionType *NewFTy);
    Function *mapKernel(ValueToValueMapTy &VMap, Module &M, Function *Kernel);

    FunctionType *mapKernelSignature(FunctionType *FTy);
    void propagateAddressSpace(Value *V);
    void mapCalls(ValueToValueMapTy &VMap, Function *F);

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
            SPIR_DEBUG(errs() << "SPIR: Running on Kernel " << kernel->getName()
                              << "\n");
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

    auto NewF = mapFunction(VMap, M, F, NewFTy);

    NewF->setCallingConv(CallingConv::SPIR_KERNEL);

    return NewF;
}

Function *SpirConvertSccPass::mapFunction(ValueToValueMapTy &VMap, Module &M,
                                          Function *F, FunctionType *NewFTy) {
    SPIR_DEBUG(errs() << "SPIR: Mapping Function " << F->getName() << "\n");
    auto SourceM = F->getParent();
    auto FTy = F->getFunctionType();

    Function* NewF;

    if (FTy != NewFTy || SourceM != &M) {
        SPIR_DEBUG(errs() << "SPIR: Change in signature or module\n");
        SPIR_DEBUG(errs() << "SPIR: from Module " << F->getParent()->getName()
                          << " to " << M.getName() << "\n");

        NewF = Function::Create(NewFTy, F->getLinkage(), "", &M);

        if (SourceM == &M) {
            NewF->takeName(F);
        } else {
            NewF->setName(F->getName());
        }

        NewF->setCallingConv(CallingConv::SPIR_FUNC);

        ValueToValueMapTy LocalVMap;

        // Add Mappings for the Arguments
        auto OldAI = F->arg_begin();
        for (auto AI = NewF->arg_begin(), AE = NewF->arg_end(); AI != AE;
             ++AI, ++OldAI) {
            LocalVMap[OldAI] = AI;
        }

        SmallVector<ReturnInst *, 8> Returns;
        CloneFunctionInto(NewF, F, LocalVMap, /*ModuleLevelChanges =*/true,
                          Returns, "", nullptr);

        // The argument types' address spaces may have changed,
        // propagate the changes to make the function valid again
        for (auto AI = NewF->arg_begin(), AE = NewF->arg_end(); AI != AE;
             ++AI) {
            propagateAddressSpace(AI);
        }
    } else {
        SPIR_DEBUG(errs() << "SPIR: Nothing to do\n");
        assert(!F->isDeclaration() && "Cannot call external Functions");
        NewF = F;
    }

    // Add the new function mapping
    VMap[F] = NewF;

    // Map over called functions
    mapCalls(VMap, NewF);

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

void SpirConvertSccPass::propagateAddressSpace(Value *V) {
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
                        assert(NewDestTy && "Null Type");
                        Cast->mutateType(NewDestTy);
                        Changed = true;
                    }
                }
            }
            // Propagate across gep instructions
            else if (auto GEP = dyn_cast<GetElementPtrInst>(Usr)) {
                auto NewTy = PointerType::get(GEP->getResultElementType(),
                                              Ty->getPointerAddressSpace());

                if (NewTy != UsrTy) {
                    assert(NewTy && "Null Type");
                    GEP->mutateType(NewTy);
                    Changed = true;
                }
            }
            // Update the function signature for CallSites
            // so we can use it later on to map over called functions
            else if (auto CS = CallSite(Usr)) {
                auto FTy = CS.getFunctionType();
                SmallVector<Type *, 3> ArgTypes(FTy->param_begin(),
                                                FTy->param_end());
                auto ArgIdx = CS.getArgumentNo(U.operator->());
                ArgTypes[ArgIdx] = Ty;
                auto NewFTy =
                    FunctionType::get(UsrTy, ArgTypes, FTy->isVarArg());

                if (FTy != NewFTy) {
                    SPIR_DEBUG(errs() << "SPIR: Updating Call Signature for "
                                      << Usr->getName() << "\n");
                    CS.mutateFunctionType(NewFTy);
                    // The Return type didn't change
                    // Changed = true;
                }
            }
            if (Changed) {
                propagateAddressSpace(Usr);
            }
        }
    }
}
void SpirConvertSccPass::mapCalls(ValueToValueMapTy &VMap, Function *F) {
    SPIR_DEBUG(errs() << "SPIR: Mapping Calls in " << F->getName() << "\n");
    for (auto BI = F->begin(), BE = F->end(); BI != BE; ++BI) {
        assert(BI->getType() && "BB without Type");
        for (auto II = BI->begin(), IE = BI->end(); II != IE; ++II) {
            if (auto CS = CallSite(II)) {
                if (auto Callee = CS.getCalledFunction()) {
                    StringRef CName;
                    CName = Callee->getName();
                    if (!CName.startswith("julia_")) {
                        continue;
                    }

                    SPIR_DEBUG(errs() << "SPIR: Mapping CallSite for "
                                      << Callee->getName() << "\n");
                    auto MappedCallee = VMap.find(Callee);
                    // Ignore any identity mappings inserted while cloning
                    if (MappedCallee != VMap.end() &&
                        MappedCallee->second != Callee) {
                        SPIR_DEBUG(errs() << "SPIR: Found in ValueMap\n");
                        Callee = dyn_cast<Function>(VMap[Callee]);
                        assert(Callee && "Mapping target is not a Function");
                    } else {
                        auto CTy = Callee->getFunctionType();
                        SmallVector<Type *, 3> ArgTypes;
                        ArgTypes.reserve(Callee->arg_size());

                        for (auto AI = CS.arg_begin(), AE = CS.arg_end();
                             AI != AE; ++AI) {
                            ArgTypes.push_back(AI->get()->getType());
                        }

                        auto NewCTy = FunctionType::get(
                            CTy->getReturnType(), ArgTypes, CTy->isVarArg());

                        Callee =
                            mapFunction(VMap, *F->getParent(), Callee, NewCTy);
                    }
                    CS.setCalledFunction(Callee);
                    CS.setCallingConv(Callee->getCallingConv());
                }
            }
        }
    }
}
