# Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.

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

# Dependencies
include ../common.mk

# Define this on the command line to change the format of perftest readouts to
# machine readable format.
NVSHMEM_MACHINE_READABLE_OUTPUT ?= 0

ifeq ($(NVSHMEM_MACHINE_READABLE_OUTPUT), 1)
TESTCUFLAGS  += -DNVSHMEM_MACHINE_READABLE_OUTPUT
endif

NVSHMEM_PERFTEST_BUILDDIR ?= $(mkfile_dir)/obj
NVSHMEM_PERFTEST_INSTALL ?= $(mkfile_dir)/perftest_install

.PHONY : default 
default : perftest

PERFCUHELPERFILES := common/utils.cu
PERFCUSRCFILES := device/coll/alltoall_bw.cu \
                  device/coll/alltoall_latency.cu \
                  device/coll/barrier_latency.cu \
                  device/coll/bcast_latency.cu \
                  device/coll/fcollect_latency.cu \
                  device/coll/reduction_latency.cu \
                  device/coll/redmaxloc_latency.cu \
                  device/coll/sync_latency.cu \
                  device/pt-to-pt/shmem_atomic_bw.cu \
                  device/pt-to-pt/shmem_atomic_latency.cu \
                  device/pt-to-pt/shmem_atomic_ping_pong_latency.cu \
                  device/pt-to-pt/shmem_get_bw.cu \
                  device/pt-to-pt/shmem_get_latency.cu \
                  device/pt-to-pt/shmem_g_bw.cu \
                  device/pt-to-pt/shmem_g_latency.cu \
                  device/pt-to-pt/shmem_put_atomic_ping_pong_latency.cu \
                  device/pt-to-pt/shmem_put_bw.cu \
                  device/pt-to-pt/shmem_put_latency.cu \
                  device/pt-to-pt/shmem_put_ping_pong_latency.cu \
                  device/pt-to-pt/shmem_put_signal_ping_pong_latency.cu \
                  device/pt-to-pt/shmem_p_bw.cu \
                  device/pt-to-pt/shmem_p_latency.cu \
                  device/pt-to-pt/shmem_p_ping_pong_latency.cu \
                  device/pt-to-pt/shmem_signal_ping_pong_latency.cu \
                  host/pt-to-pt/stream_latency.cu \
                  device/pt-to-pt/shmem_st_bw.cu
PERFCXXSRCFILES := host/coll/alltoall.cpp \
                   host/coll/alltoall_on_stream.cpp \
                   host/coll/barrier.cpp \
                   host/coll/barrier_all.cpp \
                   host/coll/barrier_all_on_stream.cpp \
                   host/coll/barrier_on_stream.cpp \
                   host/coll/broadcast.cpp \
                   host/coll/broadcast_on_stream.cpp \
                   host/coll/fcollect.cpp \
                   host/coll/fcollect_on_stream.cpp \
                   host/coll/reduction.cpp \
                   host/coll/reduction_on_stream.cpp \
                   host/coll/sync.cpp \
                   host/coll/sync_all.cpp \
                   host/coll/sync_all_on_stream.cpp \
                   host/coll/sync_on_stream.cpp \
                   host/pt-to-pt/bw.cpp \
                   host/pt-to-pt/latency.cpp \
                   host/init/malloc.cpp

LIBDIR := $(NVSHMEM_HOME)/lib

CUPERFHELPEROBJ := $(patsubst %.cu, $(NVSHMEM_PERFTEST_BUILDDIR)/%.o, $(filter %.cu, $(PERFCUHELPERFILES)))
CUPERFBIN := $(patsubst %.cu, $(NVSHMEM_PERFTEST_BUILDDIR)/%, $(filter %.cu, $(PERFCUSRCFILES)))
CXXPERFBIN := $(patsubst %.cpp, $(NVSHMEM_PERFTEST_BUILDDIR)/%, $(filter %.cpp, $(PERFCXXSRCFILES)))

perftest : $(CUPERFHELPEROBJ) $(CUPERFBIN) $(CXXPERFBIN)
 
$(NVSHMEM_PERFTEST_BUILDDIR)/common/%.o : common/%.cu
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@ 

$(NVSHMEM_PERFTEST_BUILDDIR)/device/pt-to-pt/% : device/pt-to-pt/%.cu $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(NVSHMEM_PERFTEST_BUILDDIR)/device/coll/% : device/coll/%.cu $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(NVSHMEM_PERFTEST_BUILDDIR)/host/coll/% : host/coll/%.cpp $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(NVSHMEM_PERFTEST_BUILDDIR)/host/pt-to-pt/% : host/pt-to-pt/%.cpp $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(NVSHMEM_PERFTEST_BUILDDIR)/host/pt-to-pt/% : host/pt-to-pt/%.cu $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(NVSHMEM_PERFTEST_BUILDDIR)/host/init/% : host/init/%.cpp $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

.PHONY: clean
clean :
	rm -rf $(NVSHMEM_PERFTEST_BUILDDIR)

.PHONY: uninstall
uninstall :
	rm -rf $(NVSHMEM_PERFTEST_INSTALL)

.PHONY: purge
purge : uninstall clean

.PHONY: install
install : perftest
	mkdir -p $(NVSHMEM_PERFTEST_INSTALL)
	cp -r -v $(NVSHMEM_PERFTEST_BUILDDIR)/device/ $(NVSHMEM_PERFTEST_INSTALL)/
	cp -r -v $(NVSHMEM_PERFTEST_BUILDDIR)/host/ $(NVSHMEM_PERFTEST_INSTALL)/
