## Julia with HSAIL 

Note: The original Julia README.md is now called README.orig.md

### Build Instructions
In order to build the modified Julia, you should have at least the following
in your Make.user file
    USE_SYSTEM_LLVM=0
    JULIA_BUILD_MODE=release
    LLVM_VER = svn
    LLVM_GIT_URL_LLVM = https://github.com/rollingthunder/HLC-HSAIL-Development-LLVM.git
    LLVM_GIT_VER = hsail-stable-3.7
    LLVM_GIT_URL_HSAIL_TOOLS = https://github.com/rollingthunder/HSAIL-Tools.git
    LLVM_USE_CMAKE = 1
    # If you do not have ninja, use make here
    CMAKE_GENERATOR = Ninja
    LLVM_TARGETS = X86;HSAIL
    LLVM_DEBUG=1

Due to a quirk in the current build system, you will have to build julia once with 
    LLVM_DEBUG=1
and then again with
    LLVM_DEBUG=0
without cleaning the build directory in between.

Otherwise, you might see the following error message when trying to compile a Julia kernel:
   EmitRawText called on an MCStreamer that doesn't support it,  something must not be fully mc'ized 

### Environment Setup

*HSAIL Intrinsics*

Julia needs to be able to find the HSAIL intrinsics files.
It looks for these in the environment variable
    `HSA_BUILTINS_PATH`

A copy of the intrinsics can be found in the `builtins` directory.
So you can just say
    export HSA_BUILTINS_PATH=/path/to/julia/builtins
before running Julia and it will work.

*HSA Runtime*

The same procedure is needed for the HSA runtime libraries.
These are only found, if they are in `LD_LIBRARY_PATH`, so make sure the path to them 
(usually `/opt/hsa/lib`) is in your library path.


