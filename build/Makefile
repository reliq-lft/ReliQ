# ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ                 \
  Source file: build/config.nims                                                    \
  Contact: reliq-lft@proton.me                                                      \
                                                                                    \
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>               \
                                                                                    \
  MIT License                                                                       \
                                                                                    \
  Copyright (c) 2025 reliq-lft                                                      \
                                                                                    \
                                                                                    \
  Permission is hereby granted, free of charge, to any person obtaining a copy      \
  of this software and associated documentation files (the "Software"), to deal     \
  in the Software without restriction, including without limitation the rights      \
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         \
  copies of the Software, and to permit persons to whom the Software is             \
  furnished to do so, subject to the following conditions:                          \
                                                                                    \
  The above copyright notice and this permission notice shall be included in all    \
  copies or substantial portions of the Software.                                   \
                                                                                    \
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        \
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          \
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       \
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, \
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN   \
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

NIM = external/bin/nim

# ============================================================================
# Compile-time flags (convenience wrappers for -d:Flag:value)
# ============================================================================
# Usage: make <target> DEBUG_KERNELS=1 USE_WORKGROUPS=1 VECTOR_WIDTH=16
#
# See 'make help' for full list of options
# ============================================================================

# Build FLAGS based on user-provided variables
FLAGS =

# VectorWidth: SIMD width for AoSoA vectorization (default: 8)
ifdef VECTOR_WIDTH
FLAGS += -d:VectorWidth:$(VECTOR_WIDTH)
endif

# DebugKernels: Print generated OpenCL kernel source (default: false)
ifdef DEBUG_KERNELS
FLAGS += -d:DebugKernels:true
endif

# UseWorkGroups: Force explicit work-group size = VectorWidth (default: false)
# Useful for debugging to see thread distribution via group_id, local_id
ifdef USE_WORKGROUPS
FLAGS += -d:UseWorkGroups:true
endif

# Backend selection: opencl (default), sycl, or openmp
# Usage: make tensorview BACKEND=sycl
ifeq ($(BACKEND),sycl)
FLAGS += -d:UseSycl:true
# SYCL requires C++ compilation with icpx or clang++ with -fsycl
# The SYCL wrapper library must be compiled separately:
#   icpx -fsycl -shared -fPIC -o libreliq_sycl.so src/sycl/sycl_wrapper.cpp
SYCL_WRAPPER_LIB := lib/libreliq_sycl.so
SYCL_WRAPPER_SRC := ../src/sycl/sycl_wrapper.cpp
FLAGS += --passL:-L$(PWD)/lib --passL:-lreliq_sycl --passL:-Wl,-rpath,$(PWD)/lib
else ifeq ($(BACKEND),openmp)
FLAGS += -d:UseOpenMP:true
# OpenMP requires -fopenmp flag for both compilation and linking
FLAGS += --passC:-fopenmp --passL:-fopenmp
endif

# ============================================================================
# Build targets
# ============================================================================

.PHONY: help clean sycl-lib

# Filter out reserved targets from MAKECMDGOALS for the catch-all
BUILD_TARGETS := $(filter-out help clean reliq sycl-lib,$(MAKECMDGOALS))

# SYCL wrapper library (build with Intel DPC++ or hipSYCL)
sycl-lib:
	@mkdir -p lib
	@echo "Building SYCL wrapper library..."
	@if command -v icpx >/dev/null 2>&1; then \
		icpx -fsycl -shared -fPIC -O3 -o lib/libreliq_sycl.so ../src/sycl/sycl_wrapper.cpp; \
	elif command -v clang++ >/dev/null 2>&1 && clang++ --help 2>&1 | grep -q fsycl; then \
		clang++ -fsycl -shared -fPIC -O3 -o lib/libreliq_sycl.so ../src/sycl/sycl_wrapper.cpp; \
	elif command -v syclcc >/dev/null 2>&1; then \
		syclcc --hipsycl-targets=omp -shared -fPIC -O3 -o lib/libreliq_sycl.so ../src/sycl/sycl_wrapper.cpp; \
	else \
		echo "ERROR: No SYCL compiler found. Install Intel oneAPI (icpx) or hipSYCL (syclcc)"; \
		exit 1; \
	fi
	@echo "SYCL wrapper library built: lib/libreliq_sycl.so"

help:
	@echo ""
	@echo "ReliQ Build System"
	@echo "=================="
	@echo ""
	@echo "Usage: make <target> [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  <filename>        Build src/<filename>.nim (e.g., make tensorview)"
	@echo "  sycl-lib          Build SYCL wrapper library (required for BACKEND=sycl)"
	@echo "  clean             Remove compiled binaries"
	@echo "  help              Show this help message"
	@echo ""
	@echo "Compile-time Options (set to any value to enable):"
	@echo "  VECTOR_WIDTH=N    SIMD width for AoSoA vectorization"
	@echo "                    Default: 8 (CPU), 32 (GPU)"
	@echo "                    Example: make tensorview VECTOR_WIDTH=16"
	@echo ""
	@echo "  DEBUG_KERNELS=1   Print generated OpenCL/SYCL kernel source code"
	@echo "                    Useful for inspecting kernel generation"
	@echo "                    Example: make tensorview DEBUG_KERNELS=1"
	@echo ""
	@echo "  USE_WORKGROUPS=1  Force explicit OpenCL work-group size = VectorWidth"
	@echo "                    Enables thread debugging via group_id, local_id"
	@echo "                    Note: May impact performance; for debugging only"
	@echo "                    Example: make tensorview USE_WORKGROUPS=1"
	@echo ""
	@echo "Backend Selection:"
	@echo "  BACKEND=opencl    Use OpenCL backend (default)"
	@echo "  BACKEND=sycl      Use SYCL backend (requires Intel oneAPI or hipSYCL)"
	@echo "                    First run: make sycl-lib"
	@echo "                    Then: make tensorview BACKEND=sycl"
	@echo "  BACKEND=openmp    Use OpenMP backend (CPU-only parallelization)"
	@echo "                    Uses host memory directly, no device transfers"
	@echo "                    Example: make tensorview BACKEND=openmp"
	@echo ""
	@echo "Advanced: Pass arbitrary Nim flags via ARGS:"
	@echo "  make tensorview ARGS=\"-d:MyFlag:value --hints:off\""
	@echo ""
	@echo "Kernel Debug Variables (available in 'each' loops with echo):"
	@echo "  gid           Global work item ID (0..numSites-1)"
	@echo "  g             Alias for gid"
	@echo "  lane          Lane within vector (gid mod VectorWidth)"
	@echo "  local_id      Local work item ID within work-group"
	@echo "  group_id      Work-group ID (≈ POCL pthread ID with USE_WORKGROUPS)"
	@echo "  local_size    Size of work-group"
	@echo "  numSites      Total number of lattice sites"
	@echo "  VW            Alias for VectorWidth"
	@echo ""
	@echo "Examples:"
	@echo "  make tensorview"
	@echo "  make tensorview DEBUG_KERNELS=1"
	@echo "  make tensorview USE_WORKGROUPS=1 VECTOR_WIDTH=4"
	@echo "  make sycl-lib && make tensorview BACKEND=sycl"
	@echo ""

clean:
	$(NIM) clean build/Makefile.nims $(ARGS)

# Build target: only runs if we have actual build targets
ifneq ($(BUILD_TARGETS),)
$(BUILD_TARGETS): build-impl
	@true

.PHONY: build-impl
build-impl:
	$(NIM) build build/Makefile.nims $(FLAGS) $(ARGS) $(BUILD_TARGETS)
endif