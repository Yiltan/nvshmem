#
# Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

# Compiler / Make vars
CXX ?= /usr/bin/g++
CC ?= /usr/bin/gcc
ARCH := $(shell uname -m)

#CUDA Vars
CUDA_HOME ?= /usr/local/cuda
CUDA_DRV ?= $(CUDA_HOME)/lib64/stubs
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_VERSION = $(strip $(shell $(NVCC) --version | grep release | sed 's/^.*release \([0-9]\+\.[0-9]\+\).*/\1/'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CFLAGS ?= 
CXXFLAGS ?= 
NVCUFLAGS ?=

# NVSHMEM internal features
NVSHMEM_HOME ?= /usr/local/nvshmem/
# Enable debug mode, builds library with -g
NVSHMEM_DEBUG ?= 0
NVSHMEM_VERBOSE ?= 0
NVSHMEM_DEVEL ?= 0

# whether to build with UCX support. If yes, UCX_HOME should be set
NVSHMEM_UCX_SUPPORT ?= 0
# whether to build with ibrc support.
NVSHMEM_IBRC_SUPPORT ?= 1
# whether to build with ibdevx support.
NVSHMEM_IBDEVX_SUPPORT ?= 0
# whether to build with libfabric support.
NVSHMEM_LIBFABRIC_SUPPORT ?= 0
# whether to build with GPU-initiated communication support.
NVSHMEM_IBGDA_SUPPORT ?= 0
# whether to build with GPU-initiated communication support for GPU memory only.
# host memory will not be supported but users could gain improved performance.
NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY ?= 0
# UCX install location
UCX_HOME ?= /usr/local/ucx
# libfabric installation location
LIBFABRIC_HOME ?= /usr/local/libfabric
# Whether to build with MPI support. If yes, MPI_HOME should be set
NVSHMEM_MPI_SUPPORT ?= 1
# MPI install location
MPI_HOME ?= /usr/local/ompi/
MPICC ?= $(MPI_HOME)/bin/mpicc
# Name of MPI library
NVSHMEM_LMPI ?= -lmpi
# Whether to build with SHMEM support
NVSHMEM_SHMEM_SUPPORT ?= 0
# SHMEM install location
SHMEM_HOME ?= $(MPI_HOME)
OSHCC ?= $(SHMEM_HOME)/bin/oshcc
# Name of SHMEM library
NVSHMEM_LSHMEM ?= -loshmem
# Whether to build NVSHMEM static library
NVSHMEM_TEST_STATIC_LIB ?= 0
# Whether to build the PMIX bootstrap
NVSHMEM_PMIX_SUPPORT ?= 0
NVSHMEM_ENABLE_ALL_DEVICE_INLINING ?= 0

MPI_LIBS := $(NVSHMEM_LMPI)
SHMEM_LIBS := $(NVSHMEM_LSHMEM)
LDFLAGS := -L$(CUDA_LIB) -lcudart_static -L$(CUDA_DRV) -lnvidia-ml
TESTCUFLAGS  := -dc -ccbin $(CXX) -std=c++11 -Xcompiler -fPIC
TESTLDFLAGS := -ccbin $(CXX) -std=c++11 -lcuda -L$(CUDA_LIB) -L$(CUDA_DRV) -lnvidia-ml -L$(NVSHMEM_HOME)/lib -Xlinker -rpath=$(NVSHMEM_HOME)/lib
ifeq ($(NVSHMEM_TEST_STATIC_LIB), 1)
TESTLDFLAGS += -lnvshmem
else
TESTLDFLAGS += -lnvshmem_host -lnvshmem_device
endif
TESTINC := -I$(CUDA_INC) -I$(mkfile_dir)/common

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE_DEFAULT = -gencode=arch=compute_70,code=sm_70
ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 11; echo $$?),0)
NVCC_GENCODE_DEFAULT += -gencode=arch=compute_80,code=sm_80
# The threads option was introuced to NVCC in CUDA 11.2
ifeq ($(shell test "0$(CUDA_MINOR)" -ge 2; echo $$?),0)
NVCUFLAGS += -t 4
TESTCUFLAGS += -t 4
endif
ifeq ($(shell test "0$(CUDA_MINOR)" -ge 8; echo $$?),0)
NVCC_GENCODE_DEFAULT += -gencode=arch=compute_90,code=sm_90
endif
else
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 12; echo $$?),0)
NVCC_GENCODE_DEFAULT += -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90
NVCUFLAGS += -t 4
TESTCUFLAGS += -t 4
endif
endif

NVCC_GENCODE ?= $(NVCC_GENCODE_DEFAULT)

TESTCUFLAGS += $(NVCC_GENCODE)

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
TESTINC += -I$(SHMEM_HOME)/include -DNVSHMEMTEST_SHMEM_SUPPORT
TESTLDFLAGS += -L$(SHMEM_HOME)/lib $(SHMEM_LIBS)
endif
TESTINC += -I$(NVSHMEM_HOME)/include

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
TESTINC += -I$(MPI_HOME)/include -DNVSHMEMTEST_MPI_SUPPORT
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIBS)
endif

ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
LIBFABRIC_LIBDIR = $(LIBFABRIC_HOME)/$(shell if [ -d $(LIBFABRIC_HOME)/lib ] ; then echo lib ; else echo lib64 ; fi)
endif

# NVCC doesn't support redefining -O (even with the same value) so we need to scrub it from cu flag variables.
OPTIMIZATION_ARGUMENTS= -O -O0 -O1 -O2 -O3 -Os -Ofast
TESTCUFLAGS := $(filter-out $(OPTIMIZATION_ARGUMENTS),$(TESTCUFLAGS)) 
NVCUFLAGS := $(filter-out $(OPTIMIZATION_ARGUMENTS),$(NVCUFLAGS)) 

ifeq ($(NVSHMEM_DEBUG), 0)
TESTCUFLAGS  += -O3
NVCUFLAGS += -O3
CXXFLAGS  += -O3
else
TESTCUFLAGS  += -O0 -g -G -D_NVSHMEM_DEBUG
NVCUFLAGS += -O0 -g -G -DNVSHMEM_TRACE -D_NVSHMEM_DEBUG
CXXFLAGS  += -O0 -g -DNVSHMEM_TRACE -D_NVSHMEM_DEBUG
endif

ifneq ($(NVSHMEM_VERBOSE), 0)
TESTCUFLAGS  += -Xptxas -v
NVCUFLAGS += -Xptxas -v
endif

#TODO: add -Werror and -Wall to TESTCUFLAGS. Have to fix all warnings first.
ifneq ($(NVSHMEM_DEVEL), 0)
TESTCUFLAGS += -Werror all-warnings -Xcompiler -Wall,-Wextra,-Wno-unused-function,-Wno-unused-parameter
NVCUFLAGS += -Werror all-warnings -Xcompiler -Wall,-Wextra,-Wno-unused-function,-Wno-unused-parameter
CXXFLAGS  += -Werror -Wall -Wextra -Wno-unused-function -Wno-unused-parameter
endif
