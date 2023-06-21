# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

# Define this variable for the Include Variable in common.mk
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

include ../common.mk

NVSHMEM_EXAMPLES_BUILDDIR ?= $(mkfile_dir)/obj
NVSHMEM_EXAMPLES_INSTALL ?= $(mkfile_dir)/examples_install
 
.PHONY : default 
default : examples

EXAMPLECUSRCFILES := \
    collective-launch.cu        \
    on-stream.cu                \
    thread-group.cu             \
    put-block.cu                \
    dev-guide-ring.cu           \
    ring-bcast.cu

EXAMPLECXXSRCFILES := \
    hello.cpp

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
EXAMPLECUSRCFILES += mpi-based-init.cu dev-guide-ring-mpi.cu
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
EXAMPLECUSRCFILES += shmem-based-init.cu
endif

CUPERFBIN   := $(patsubst %.cu, $(NVSHMEM_EXAMPLES_BUILDDIR)/%, $(filter %.cu, $(EXAMPLECUSRCFILES)))
CXXPERFBIN  := $(patsubst %.cpp, $(NVSHMEM_EXAMPLES_BUILDDIR)/%, $(filter %.cpp, $(EXAMPLECXXSRCFILES)))

examples : $(CUPERFBIN) $(CXXPERFBIN)
 
$(NVSHMEM_EXAMPLES_BUILDDIR)/% : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(TESTCPPHELPEROBJ) $(TESTCUHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(NVSHMEM_EXAMPLES_BUILDDIR)/% : %.cpp
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(TESTCPPHELPEROBJ) $(TESTCUHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

.PHONY : clean 
clean :
	rm -rf $(NVSHMEM_EXAMPLES_BUILDDIR)

.PHONY : uninstall 
uninstall : 
	rm -rf $(NVSHMEM_EXAMPLES_INSTALL)

.PHONY : purge 
purge : clean uninstall

.PHONY : install 
install : examples
	mkdir -p $(NVSHMEM_EXAMPLES_INSTALL)
	cp -r -v $(NVSHMEM_EXAMPLES_BUILDDIR)/* $(NVSHMEM_EXAMPLES_INSTALL)/
