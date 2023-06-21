#
# Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information
#

# Define this variable for the Include Variable in common.mk
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

# External dependencies
include common.mk
include version.mk

# Build/install location
# location where the build will be installed
NVSHMEM_PREFIX ?= /usr/local/nvshmem/
# build location
NVSHMEM_BUILDDIR ?= $(abspath build)
# MPI/SHMEM Support

# GDRCopy install/headers location
GDRCOPY_HOME ?= /usr/local/gdrdrv
# NCCL install/headers location
NCCL_HOME ?= /usr/local/nccl
NVSHMEM_USE_NCCL ?= 0
# Whether to build with MPI support. If yes, MPI_HOME should be set
NVSHMEM_USE_GDRCOPY ?= 1
# Include support for PMIx as the process manager interface
PMIX_HOME ?= /usr
# One of the below can be set to 1 to override the default PMI
NVSHMEM_DEFAULT_PMIX ?= 0
NVSHMEM_DEFAULT_PMI2 ?= 0

# This can be set to override the default remote transport.
NVSHMEM_DEFAULT_UCX ?= 0

# NVSHMEM internal features
NVSHMEM_TRACE ?= 0
NVSHMEM_NVTX ?= 1

NVSHMEM_DISABLE_COLL_POLL ?= 1
NVSHMEM_GPU_COLL_USE_LDST ?= 0
# Timeout if stuck for long in wait loops in device
NVSHMEM_TIMEOUT_DEVICE_POLLING ?= 0
# Use dlmalloc (instead of custom_malloc) as heap
# allocator (will work only if not using CUDA VMM)
NVSHMEM_USE_DLMALLOC ?= 0

BUILD_OPTIONS_STR = "\\n"
ifeq ($(ARCH), x86_64)
CXXFLAGS += -fPIC -I$(CUDA_INC) -msse
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_X86_64\\n"
else
ifeq ($(ARCH), ppc64le)
CXXFLAGS   += -fPIC -I$(CUDA_INC) -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_PPC64LE\\n"
endif
endif
NVCUFLAGS  += -Xcompiler -fPIC -ccbin $(CXX) $(NVCC_GENCODE) -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS

ifneq ($(NVSHMEM_PMIX_SUPPORT), 1)
# Don't allow PMIX to be set as default unless it's in the build
NVSHMEM_DEFAULT_PMIX := 0
endif

ifeq ($(NVSHMEM_ENV_ALL), 1)
CXXFLAGS  += -DNVSHMEM_ENV_ALL
endif

ifeq ($(NVSHMEM_DEFAULT_PMIX), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_DEFAULT_PMIX\\n"
else
ifeq ($(NVSHMEM_DEFAULT_PMI2), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_DEFAULT_PMI2\\n"
endif
endif

ifeq ($(NVSHMEM_DEFAULT_UCX), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_DEFAULT_UCX\\n"
endif

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
CXXFLAGS  += -I$(MPI_HOME)/include
NVCUFLAGS += -I$(MPI_HOME)/include
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_MPI_SUPPORT\\n"
endif

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
CXXFLAGS  += -I$(UCX_HOME)/include
NVCUFLAGS += -I$(UCX_HOME)/include
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_UCX_SUPPORT\\n"
endif

ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
CXXFLAGS  += -I$(LIBFABRIC_HOME)/include
NVCUFLAGS += -I$(LIBFABRIC_HOME)/include
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_LIBFABRIC_SUPPORT\\n"
endif

ifeq ($(NVSHMEM_IBRC_SUPPORT), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_IBRC_SUPPORT\\n"
endif

ifeq ($(NVSHMEM_IBDEVX_SUPPORT), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_IBDEVX_SUPPORT\\n"
endif

ifeq ($(NVSHMEM_IBGDA_SUPPORT), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_IBGDA_SUPPORT\\n"
ifeq ($(NVSHMEM_DEBUG), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_IBGDA_DEBUG\\n"
endif
ifeq ($(NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY\\n"
endif
endif

ifeq ($(NVSHMEM_USE_GDRCOPY), 1)
ifneq ("$(wildcard $(mkfile_dir)/include_gdrcopy)","")
CXXFLAGS  += -I$(mkfile_dir)/include_gdrcopy
NVCUFLAGS += -I$(mkfile_dir)/include_gdrcopy
else
CXXFLAGS  += -I$(GDRCOPY_HOME)/include
NVCUFLAGS += -I$(GDRCOPY_HOME)/include
endif
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_USE_GDRCOPY\\n"
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
CXXFLAGS  += -I$(SHMEM_HOME)/include
NVCUFLAGS += -I$(SHMEM_HOME)/include
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_SHMEM_SUPPORT\\n"
endif

# If we have an internal NCCL header, use it. Otherwise, use the one in NCCL_HOME
ifeq ($(NVSHMEM_USE_NCCL), 1)
ifneq ("$(wildcard $(mkfile_dir)/include_nccl)","")
CXXFLAGS  += -I$(mkfile_dir)/include_nccl
NVCUFLAGS += -I$(mkfile_dir)/include_nccl
else
CXXFLAGS  += -I$(NCCL_HOME)/include
NVCUFLAGS += -I$(NCCL_HOME)/include
endif
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_USE_NCCL\\n"
endif

ifeq ($(NVSHMEM_DISABLE_COLL_POLL), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_DISABLE_COLL_POLL\\n"
endif

ifeq ($(NVSHMEM_GPU_COLL_USE_LDST), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_GPU_COLL_USE_LDST\\n"
endif

ifeq ($(NVSHMEM_TIMEOUT_DEVICE_POLLING), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_TIMEOUT_DEVICE_POLLING\\n"
endif

ifeq ($(NVSHMEM_TRACE), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_TRACE\\n"
endif

ifeq ($(NVSHMEM_ENABLE_ALL_DEVICE_INLINING), 1)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVSHMEM_ENABLE_ALL_DEVICE_INLINING\\n"
endif

# ignore the following for clean targets
ifeq (,$(findstring $(MAKECMDGOALS),purge clean))
ifeq ($(NVSHMEM_NVTX), 0)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVTX_DISABLE\\n"
else
# C++11 is required for NVTX support
cppver := $(shell sh ./scripts/test_cxx11.sh $(CXX) "$(CXXFLAGS)")
ifneq ($(cppver),)
BUILD_OPTIONS_STR:=${BUILD_OPTIONS_STR}"\#define NVTX_DISABLE\\n"
$(info ${cppver})
endif
endif
endif

NVCU_MAXRREGCOUNT := --maxrregcount 32

NVSHMEM_INFO_BUILD_VARS ?= 1

ifeq ($(NVSHMEM_INFO_BUILD_VARS), 1)
INFO_BUILD_VARS := $(strip $(foreach v, $(sort $(filter NVSHMEM_% %_HOME, $(.VARIABLES))), $(v)=\\\"$($(v))\\\"))
else
INFO_BUILD_VARS :=
endif

.PHONY : default 
default : lib

LICENSE_FILES := NVSHMEM-SLA.txt COPYRIGHT.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(NVSHMEM_BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${NVSHMEM_BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${NVSHMEM_BUILDDIR}
	cp $< $@

INCEXPORTS_NVSHMEM  := nvshmem.h nvshmemx.h
INCEXPORTS := nvshmem.h \
              nvshmem_api.h \
              nvshmem_bootstrap_defines.h \
              nvshmem_bootstrap.h \
              nvshmem_coll_api.h \
              nvshmem_common.cuh \
              nvshmem_constants.h \
              nvshmem_defines.h \
              nvshmem_types.h \
              nvshmemi_util.h \
              nvshmemi_team.h \
              nvshmemx.h \
              nvshmemx_api.h \
              nvshmemx_coll_api.h \
              nvshmemi_constants.h \
              nvshmemi_transport_defines.h \
              nvshmemx_defines.h \
              nvshmemx_error.h \
              nvshmemi_proxy.h \
              nvshmemi_ibgda.h \
              device/pt-to-pt/ibgda_device.cuh \
              device/pt-to-pt/proxy_device.cuh \
              device/pt-to-pt/utils_device.h \
              device/coll/defines.cuh \
              device/coll/utils.cuh \
              device/coll/alltoall.cuh \
              device/coll/broadcast.cuh \
              device/coll/fcollect.cuh \
              device/coll/barrier.cuh \
              device/coll/reduce.cuh \
              device/team/team_device.cuh \
              device/init/query_device.cuh

PLUGINEXPORTS := src/bootstrap/bootstrap_pmix.c \
                 src/bootstrap/bootstrap_pmi.cpp \
                 src/bootstrap/bootstrap_mpi.c \
                 src/bootstrap/bootstrap_util.h \
                 src/bootstrap/bootstrap_shmem.c

TRANSPORTINCEXPORTS := transport.h \
                       env_defs.h  \
                       cudawrap.h  \

TRANSPORTEXPORTS := common/mlx5_ifc.h              \
                    common/mlx5_prm.h              \
                    common/transport_common.h      \
                    common/transport_ib_common.h   \
                    common/transport_common.cpp    \
                    common/transport_ib_common.cpp \
                    ibdevx/ibdevx.cpp              \
                    ibdevx/ibdevx.h                \
                    ibgda/ibgda.cpp                \
                    ibrc/ibrc.cpp                  \
                    libfabric/libfabric.cpp        \
                    libfabric/libfabric.h          \
                    ucx/ucx.cpp                    \
                    ucx/ucx.h                      \

HOSTLIBSRCFILES := bootstrap/bootstrap.cpp \
               bootstrap/bootstrap_loader.cpp

ifeq ($(NVSHMEM_IBGDA_SUPPORT), 1)
HOSTLIBSRCFILES += init/ibgda_init.cu
DEVICELIBSRCFILES += init/ibgda_init_device.cu
endif

HOSTLIBSRCFILES += coll/host/cpu_coll.cpp \
                   coll/host/alltoall.cpp \
                   coll/host/alltoall_on_stream.cpp \
                   coll/host/barrier.cpp \
                   coll/host/barrier_on_stream.cpp \
                   coll/host/broadcast.cpp \
                   coll/host/broadcast_on_stream.cpp \
                   coll/host/fcollect.cpp \
                   coll/host/fcollect_on_stream.cpp \
                   coll/host/rdxn.cpp \
                   coll/host/rdxn_on_stream.cpp \
                   comm/host/putget.cpp \
                   comm/host/fence.cpp \
                   comm/proxy/proxy.cpp \
                   comm/host/quiet.cpp \
                   comm/host/sync.cpp \
                   comm/host/amo.cpp \
                   comm/transport.cpp \
                   comm/transports/p2p/p2p.cpp \
                   init/init.cu \
                   init/cudawrap.cpp \
                   init/init_nvtx.cpp \
                   init/query_host.cpp \
                   launch/collective_launch.cpp \
                   mem/mem.cpp \
                   team/team.cu \
                   team/team_internal.cu \
                   topo/topo.cpp \
                   util/cs.cpp \
                   util/debug.cpp \
                   util/env_vars.cpp \
                   util/util.cpp \
                   util/sockets.cpp

DEVICELIBSRCFILES += coll/device/gpu_coll.cu \
                     coll/device/recexchalgo.cu \
                     comm/device/proxy_device.cu \
                     launch/collective_launch_device.cu \
                     init/init_device.cu

ifeq ($(NVSHMEM_ENABLE_ALL_DEVICE_INLINING), 1)
INCEXPORTS += device/pt-to-pt/transfer_device.cuh
else
INCEXPORTS += device/pt-to-pt/nvshmemi_transfer_api.cuh
DEVICELIBSRCFILES += comm/device/transfer_device.cu
endif


ifeq ($(NVSHMEM_USE_DLMALLOC), 1)
HOSTLIBSRCFILES += mem/dlmalloc.cpp
else
HOSTLIBSRCFILES += mem/custom_malloc.cpp
endif

DEVICELIBSRCFILES_NOMAXRREGCOUNT = \
               coll/device/kernels/alltoall.cu \
               coll/device/kernels/barrier.cu \
               coll/device/kernels/broadcast.cu \
               coll/device/kernels/fcollect.cu \
               coll/device/kernels/reduce.cu \
               comm/host/cuda_interface_sync.cu \
               comm/host/proxy/rma.cu \
               comm/host/quiet_on_stream.cu

HOSTLIBNAME   := libnvshmem_host.so
DEVICELIBNAME := libnvshmem_device.a
LIBNAME       := libnvshmem.a

INCDIR := $(NVSHMEM_BUILDDIR)/include
LIBDIR := $(NVSHMEM_BUILDDIR)/lib
BINDIR := $(NVSHMEM_BUILDDIR)/bin
PLUGINSDIR := $(NVSHMEM_BUILDDIR)/share/nvshmem/src/bootstrap-plugins
TRANSPORTSDIR := $(NVSHMEM_BUILDDIR)/share/nvshmem/src/transport-plugins
OBJDIR_NVSHMEM := $(NVSHMEM_BUILDDIR)/obj_nvshmem

TRANSPORT_INCDIR = src/comm/transports/common
TRANSPORT_OBJDIR_NVSHMEM := $(NVSHMEM_BUILDDIR)/obj_nvshmem/transport

BUILT_HEADERS := $(INCDIR)/nvshmem_version.h $(INCDIR)/nvshmem_build_options.h

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))

HOSTLIBSONAME   := $(HOSTLIBNAME:%=%.$(NVSHMEM_MAJOR))
HOSTLIBTARGET   := $(HOSTLIBNAME:%=%.$(NVSHMEM_MAJOR).$(NVSHMEM_MINOR).$(NVSHMEM_PATCH))
DEVICELIBTARGET := $(DEVICELIBNAME)
LIBTARGET       := $(LIBNAME)
HOSTLIBOBJ      := $(patsubst %.cpp, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cpp, $(HOSTLIBSRCFILES)))
HOSTLIBOBJ      += $(patsubst %.c, $(OBJDIR_NVSHMEM)/%.o, $(filter %.c, $(HOSTLIBSRCFILES)))
HOSTLIBOBJ      += $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(HOSTLIBSRCFILES)))
DEVICELIBOBJ    := $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(DEVICELIBSRCFILES)))
DEVICELIBOBJ_NOMAXRREGCOUNT = $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(DEVICELIBSRCFILES_NOMAXRREGCOUNT)))
DEVICELIBOBJ_NOMAXRREGCOUNT += $(patsubst %.cpp, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cpp, $(DEVICELIBSRCFILES_NOMAXRREGCOUNT)))

LIBINC     := -Isrc/include -Isrc/util -Isrc/bootstrap -Isrc/coll/host -Isrc/coll/device -Isrc/coll -Isrc/topo
LIBINC     += -Isrc/pmi/pmi-2 -Isrc/pmi/simple-pmi -I$(INCDIR)

ifeq ($(NVSHMEM_USE_GDRCOPY), 1)
TRANSPORT_GDR_HELPER_FILES = src/comm/transports/common/transport_gdr_common.cpp
TRANSPORT_GDR_OUTPUT_FILES = $(TRANSPORT_OBJDIR_NVSHMEM)/transport_gdr_common.o
else 
TRANSPORT_GDR_HELPER_FILES =
TRANSPORT_GDR_OUTPUT_FILES =
endif

TRANSPORTS =

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
UCX_TRANSPORT_HELPER_FILES = src/comm/transports/common/transport_common.cpp $(TRANSPORT_GDR_HELPER_FILES)
UCX_TRANSPORT_OUTPUT_FILES = $(TRANSPORT_OBJDIR_NVSHMEM)/transport_common.o $(TRANSPORT_GDR_OUTPUT_FILES)
UCX_TRANSPORT_REALNAME := nvshmem_transport_ucx.so.$(TRANSPORT_VERSION_MAJOR).$(TRANSPORT_VERSION_MINOR).$(TRANSPORT_VERSION_MINOR)
UCX_TRANSPORT_SONAME := nvshmem_transport_ucx.so.$(TRANSPORT_VERSION_MAJOR)
UCX_TRANSPORT := nvshmem_transport_ucx.so
TRANSPORTS += $(LIBDIR)/$(UCX_TRANSPORT_REALNAME)
endif
ifeq ($(NVSHMEM_IBRC_SUPPORT), 1)
IBRC_TRANSPORT_REALNAME := nvshmem_transport_ibrc.so.$(TRANSPORT_VERSION_MAJOR).$(TRANSPORT_VERSION_MINOR).$(TRANSPORT_VERSION_MINOR)
IBRC_TRANSPORT_SONAME := nvshmem_transport_ibrc.so.$(TRANSPORT_VERSION_MAJOR)
IBRC_TRANSPORT := nvshmem_transport_ibrc.so
TRANSPORTS += $(LIBDIR)/$(IBRC_TRANSPORT_REALNAME)
IBRC_TRANSPORT_HELPER_FILES = src/comm/transports/common/transport_common.cpp $(TRANSPORT_GDR_HELPER_FILES) src/comm/transports/common/transport_ib_common.cpp
IBRC_TRANSPORT_OUTPUT_FILES = $(TRANSPORT_OBJDIR_NVSHMEM)/transport_common.o $(TRANSPORT_GDR_OUTPUT_FILES) $(TRANSPORT_OBJDIR_NVSHMEM)/transport_ib_common.o
endif
ifeq ($(NVSHMEM_IBDEVX_SUPPORT), 1)
IBDEVX_TRANSPORT_HELPER_FILES = src/comm/transports/common/transport_common.cpp src/comm/transports/common/transport_ib_common.cpp src/comm/transports/common/transport_mlx5_common.cpp
IBDEVX_TRANSPORT_OUTPUT_FILES = $(TRANSPORT_OBJDIR_NVSHMEM)/transport_common.o $(TRANSPORT_OBJDIR_NVSHMEM)/transport_ib_common.o $(TRANSPORT_OBJDIR_NVSHMEM)/transport_mlx5_common.o
IBDEVX_TRANSPORT_REALNAME := nvshmem_transport_ibdevx.so.$(TRANSPORT_VERSION_MAJOR).$(TRANSPORT_VERSION_MINOR).$(TRANSPORT_VERSION_MINOR)
IBDEVX_TRANSPORT_SONAME := nvshmem_transport_ibdevx.so.$(TRANSPORT_VERSION_MAJOR)
IBDEVX_TRANSPORT := nvshmem_transport_ibdevx.so
TRANSPORTS += $(LIBDIR)/$(IBDEVX_TRANSPORT_REALNAME)
endif
ifeq ($(NVSHMEM_IBGDA_SUPPORT), 1)
IBGDA_TRANSPORT_HELPER_FILES = src/comm/transports/common/transport_common.cpp src/comm/transports/common/transport_ib_common.cpp src/comm/transports/common/transport_mlx5_common.cpp
IBGDA_TRANSPORT_OUTPUT_FILES = $(TRANSPORT_OBJDIR_NVSHMEM)/transport_common.o $(TRANSPORT_OBJDIR_NVSHMEM)/transport_ib_common.o $(TRANSPORT_OBJDIR_NVSHMEM)/transport_mlx5_common.o
IBGDA_TRANSPORT_REALNAME := nvshmem_transport_ibgda.so.$(TRANSPORT_VERSION_MAJOR).$(TRANSPORT_VERSION_MINOR).$(TRANSPORT_VERSION_MINOR)
IBGDA_TRANSPORT_SONAME := nvshmem_transport_ibgda.so.$(TRANSPORT_VERSION_MAJOR)
IBGDA_TRANSPORT := nvshmem_transport_ibgda.so
TRANSPORTS += $(LIBDIR)/$(IBGDA_TRANSPORT_REALNAME)
endif
ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
LIBFABRIC_TRANSPORT_HELPER_FILES = src/comm/transports/common/transport_common.cpp
LIBFABRIC_TRANSPORT_OUTPUT_FILES = $(TRANSPORT_OBJDIR_NVSHMEM)/transport_common.o
LIBFABRIC_TRANSPORT_REALNAME := nvshmem_transport_libfabric.so.$(TRANSPORT_VERSION_MAJOR).$(TRANSPORT_VERSION_MINOR).$(TRANSPORT_VERSION_MINOR)
LIBFABRIC_TRANSPORT_SONAME := nvshmem_transport_libfabric.so.$(TRANSPORT_VERSION_MAJOR)
LIBFABRIC_TRANSPORT := nvshmem_transport_libfabric.so
TRANSPORTS += $(LIBDIR)/$(LIBFABRIC_TRANSPORT_REALNAME)
endif

PLUGINEXPORTTARGETS := $(addprefix $(PLUGINSDIR)/, $(notdir $(PLUGINEXPORTS)))
TRANSPORTEXPORTTARGETS := $(addprefix $(TRANSPORTSDIR)/, $(TRANSPORTEXPORTS))
TRANSPORTINCEXPORTTARGETS := $(addprefix $(TRANSPORTSDIR)/common/, $(TRANSPORTINCEXPORTS))

PMI_PLUGIN := nvshmem_bootstrap_pmi.so
PMI_PLUGIN_SONAME := $(PMI_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR))
PMI_PLUGIN_TARGET   := $(PMI_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR).$(BOOTSTRAP_VERSION_MINOR).$(BOOTSTRAP_VERSION_PATCH))

PMI2_PLUGIN := nvshmem_bootstrap_pmi2.so
PMI2_PLUGIN_SONAME := $(PMI2_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR))
PMI2_PLUGIN_TARGET   := $(PMI2_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR).$(BOOTSTRAP_VERSION_MINOR).$(BOOTSTRAP_VERSION_PATCH))

PLUGINS    := $(LIBDIR)/$(PMI_PLUGIN_TARGET) $(LIBDIR)/$(PMI2_PLUGIN_TARGET)
ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
PMIX_PLUGIN := nvshmem_bootstrap_pmix.so
PMIX_PLUGIN_SONAME := $(PMIX_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR))
PMIX_PLUGIN_TARGET   := $(PMIX_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR).$(BOOTSTRAP_VERSION_MINOR).$(BOOTSTRAP_VERSION_PATCH))
PLUGINS    += $(LIBDIR)/$(PMIX_PLUGIN_TARGET)
endif
ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
MPI_PLUGIN := nvshmem_bootstrap_mpi.so
MPI_PLUGIN_SONAME := $(MPI_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR))
MPI_PLUGIN_TARGET   := $(MPI_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR).$(BOOTSTRAP_VERSION_MINOR).$(BOOTSTRAP_VERSION_PATCH))
PLUGINS    += $(LIBDIR)/$(MPI_PLUGIN_TARGET)
endif
ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
SHMEM_PLUGIN := nvshmem_bootstrap_shmem.so
SHMEM_PLUGIN_SONAME := $(SHMEM_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR))
SHMEM_PLUGIN_TARGET   := $(SHMEM_PLUGIN:%=%.$(BOOTSTRAP_VERSION_MAJOR).$(BOOTSTRAP_VERSION_MINOR).$(BOOTSTRAP_VERSION_PATCH))
PLUGINS    += $(LIBDIR)/$(SHMEM_PLUGIN_TARGET)
endif

.PHONY: lib 
lib : $(INCTARGETS) $(LIBDIR)/$(DEVICELIBTARGET) $(LIBDIR)/$(HOSTLIBTARGET) $(LIBDIR)/$(LIBTARGET) $(PLUGINS) $(PLUGINEXPORTTARGETS) $(TRANSPORTEXPORTTARGETS) $(TRANSPORTINCEXPORTTARGETS) $(TRANSPORTS) $(BINDIR)/nvshmem-info

EXTRA_NVCUFLAGS = $(NVCU_MAXRREGCOUNT)
$(DEVICELIBOBJ_NOMAXRREGCOUNT) : EXTRA_NVCUFLAGS =

$(LIBDIR)/$(LIBTARGET) : $(HOSTLIBOBJ)  $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT)
	@mkdir -p $(LIBDIR)
	$(NVCC) -lib -o $@ $(HOSTLIBOBJ) $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT) 

$(LIBDIR)/$(HOSTLIBTARGET) : $(HOSTLIBOBJ) $(LIBDIR)/$(DEVICELIBTARGET) nvshmem_host.sym
	@mkdir -p $(LIBDIR)
	$(NVCC) $(NVCC_GENCODE) -shared -Xlinker --version-script=nvshmem_host.sym -Xlinker --no-as-needed -Xlinker -soname,$(HOSTLIBSONAME) -o $@ $(HOSTLIBOBJ) $(LIBDIR)/$(DEVICELIBTARGET) $(LDFLAGS)
	ln -sf $(HOSTLIBSONAME) $(LIBDIR)/$(HOSTLIBNAME)
	ln -sf $(HOSTLIBTARGET) $(LIBDIR)/$(HOSTLIBSONAME)

$(LIBDIR)/$(DEVICELIBTARGET) : $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT)
	@mkdir -p $(LIBDIR)
	$(NVCC) -lib -o $@ $(DEVICELIBOBJ) $(DEVICELIBOBJ_NOMAXRREGCOUNT)

$(PLUGINSDIR)/%: src/bootstrap/%
	@mkdir -p $(PLUGINSDIR)
	cp -f $< $@

$(TRANSPORTSDIR)/common/%: src/comm/transports/common/%
	@mkdir -p $(TRANSPORTSDIR)/common
	cp -f $< $@

$(TRANSPORTSDIR)/common/%: src/include/%
	@mkdir -p $(TRANSPORTSDIR)/common
	cp -f $< $@

$(TRANSPORTSDIR)/ibgda/%: src/comm/transports/ibgda/%
	@mkdir -p $(TRANSPORTSDIR)/ibgda
	cp -f $< $@

$(TRANSPORTSDIR)/ibdevx/%: src/comm/transports/ibdevx/%
	@mkdir -p $(TRANSPORTSDIR)/ibdevx
	cp -f $< $@

$(TRANSPORTSDIR)/ibrc/%: src/comm/transports/ibrc/%
	@mkdir -p $(TRANSPORTSDIR)/ibrc
	cp -f $< $@

$(TRANSPORTSDIR)/libfabric/%: src/comm/transports/libfabric/%
	@mkdir -p $(TRANSPORTSDIR)/libfabric
	cp -f $< $@

$(TRANSPORTSDIR)/ucx/%: src/comm/transports/ucx/%
	@mkdir -p $(TRANSPORTSDIR)/ucx
	cp -f $< $@

$(LIBDIR)/$(PMI_PLUGIN_TARGET): src/bootstrap/bootstrap_pmi.cpp $(BUILT_HEADERS) $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmi.o $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmiutil.o
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(PMI_PLUGIN_TARGET) -fpic -I$(INCDIR) -Isrc/pmi/simple-pmi $< -o $@ $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmi.o $(OBJDIR_NVSHMEM)/pmi/simple-pmi/simple_pmiutil.o
	ln -sf $(PMI_PLUGIN_SONAME) $(LIBDIR)/$(PMI_PLUGIN)
	ln -sf $(PMI_PLUGIN_TARGET) $(LIBDIR)/$(PMI_PLUGIN_SONAME)

$(LIBDIR)/$(PMI2_PLUGIN_TARGET): src/bootstrap/bootstrap_pmi.cpp $(BUILT_HEADERS) $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_api.o $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_util.o
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(PMI2_PLUGIN_TARGET) -fpic -Xlinker --version-script=nvshmem_bootstrap.sym -DNVSHMEM_BUILD_PMI2 -I$(INCDIR) -Isrc/pmi/pmi-2 $< -o $@ $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_api.o $(OBJDIR_NVSHMEM)/pmi/pmi-2/pmi2_util.o
	ln -sf $(PMI2_PLUGIN_SONAME) $(LIBDIR)/$(PMI2_PLUGIN)
	ln -sf $(PMI2_PLUGIN_TARGET) $(LIBDIR)/$(PMI2_PLUGIN_SONAME)

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
$(LIBDIR)/$(PMIX_PLUGIN_TARGET): src/bootstrap/bootstrap_pmix.c $(BUILT_HEADERS)
	@mkdir -p $(LIBDIR)
	$(CC) $(CFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(PMIX_PLUGIN_TARGET) -fpic -I$(INCDIR) -I$(PMIX_HOME)/include $< -L$(PMIX_HOME)/lib -lpmix -o $@
	ln -sf $(PMIX_PLUGIN_SONAME) $(LIBDIR)/$(PMIX_PLUGIN)
	ln -sf $(PMIX_PLUGIN_TARGET) $(LIBDIR)/$(PMIX_PLUGIN_SONAME)
endif

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
$(LIBDIR)/$(MPI_PLUGIN_TARGET): src/bootstrap/bootstrap_mpi.c $(BUILT_HEADERS)
	@mkdir -p $(LIBDIR)
	$(MPICC) $(CFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(MPI_PLUGIN_TARGET) -fpic -I$(INCDIR) -Isrc/include -Isrc/bootstrap $< -o $@
	ln -sf $(MPI_PLUGIN_SONAME) $(LIBDIR)/$(MPI_PLUGIN)
	ln -sf $(MPI_PLUGIN_TARGET) $(LIBDIR)/$(MPI_PLUGIN_SONAME)
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
$(LIBDIR)/$(SHMEM_PLUGIN_TARGET): src/bootstrap/bootstrap_shmem.c $(BUILT_HEADERS)
	@mkdir -p $(LIBDIR)
	$(OSHCC) $(CFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(SHMEM_PLUGIN_TARGET) -fpic -I$(INCDIR) -Isrc/include -Isrc/bootstrap $< -o $@
	ln -sf $(SHMEM_PLUGIN_SONAME) $(LIBDIR)/$(SHMEM_PLUGIN)
	ln -sf $(SHMEM_PLUGIN_TARGET) $(LIBDIR)/$(SHMEM_PLUGIN_SONAME)
endif

$(TRANSPORT_OBJDIR_NVSHMEM)/%.o: src/comm/transports/common/%.cpp
	@mkdir -p $(TRANSPORT_OBJDIR_NVSHMEM)
	$(CXX) -c -I$(INCDIR) -Isrc/include $(CXXFLAGS) $< -o $@

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
$(LIBDIR)/$(UCX_TRANSPORT_REALNAME): src/comm/transports/ucx/ucx.cpp $(UCX_TRANSPORT_OUTPUT_FILES)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(UCX_TRANSPORT_SONAME) -fpic -Xlinker --version-script=nvshmem_transport.sym -I$(TRANSPORT_INCDIR) -I$(INCDIR) -Isrc/include $< -o $@ $(UCX_TRANSPORT_OUTPUT_FILES) -L$(CUDA_LIB) -lcudart_static -L$(UCX_HOME)/lib -lucs -lucp
	ln -sf $(UCX_TRANSPORT_REALNAME) $(LIBDIR)/$(UCX_TRANSPORT_SONAME)
	ln -sf $(UCX_TRANSPORT_SONAME) $(LIBDIR)/$(UCX_TRANSPORT)
endif
ifeq ($(NVSHMEM_IBRC_SUPPORT), 1)
$(LIBDIR)/$(IBRC_TRANSPORT_REALNAME): src/comm/transports/ibrc/ibrc.cpp $(IBRC_TRANSPORT_OUTPUT_FILES)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(IBRC_TRANSPORT_SONAME) -fpic -Xlinker --version-script=nvshmem_transport.sym -I$(TRANSPORT_INCDIR) -I$(INCDIR) -Isrc/include $< -o $@ $(IBRC_TRANSPORT_OUTPUT_FILES) -L$(CUDA_LIB) -lcudart_static
	ln -sf $(IBRC_TRANSPORT_REALNAME) $(LIBDIR)/$(IBRC_TRANSPORT_SONAME)
	ln -sf $(IBRC_TRANSPORT_SONAME) $(LIBDIR)/$(IBRC_TRANSPORT)
endif
ifeq ($(NVSHMEM_IBDEVX_SUPPORT), 1)
$(LIBDIR)/$(IBDEVX_TRANSPORT_REALNAME): src/comm/transports/ibdevx/ibdevx.cpp $(IBDEVX_TRANSPORT_OUTPUT_FILES)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(IBDEVX_TRANSPORT_SONAME) -fpic -Xlinker --version-script=nvshmem_transport.sym -I$(TRANSPORT_INCDIR) -I$(INCDIR) -Isrc/include $< -o $@ $(IBDEVX_TRANSPORT_OUTPUT_FILES) -L$(CUDA_LIB) -lcudart_static -lmlx5
	ln -sf $(IBDEVX_TRANSPORT_REALNAME) $(LIBDIR)/$(IBDEVX_TRANSPORT_SONAME)
	ln -sf $(IBDEVX_TRANSPORT_SONAME) $(LIBDIR)/$(IBDEVX_TRANSPORT)
endif
ifeq ($(NVSHMEM_IBGDA_SUPPORT), 1)
$(LIBDIR)/$(IBGDA_TRANSPORT_REALNAME): src/comm/transports/ibgda/ibgda.cpp $(IBGDA_TRANSPORT_OUTPUT_FILES)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(IBGDA_TRANSPORT_SONAME) -fpic -Xlinker --version-script=nvshmem_transport.sym -I$(TRANSPORT_INCDIR) -I$(INCDIR) -Isrc/include $< -o $@ $(IBGDA_TRANSPORT_OUTPUT_FILES) -L$(CUDA_LIB) -lcudart_static -lmlx5
	ln -sf $(IBGDA_TRANSPORT_REALNAME) $(LIBDIR)/$(IBGDA_TRANSPORT_SONAME)
	ln -sf $(IBGDA_TRANSPORT_SONAME) $(LIBDIR)/$(IBGDA_TRANSPORT)
endif
ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
$(LIBDIR)/$(LIBFABRIC_TRANSPORT_REALNAME): src/comm/transports/libfabric/libfabric.cpp $(LIBFABRIC_TRANSPORT_OUTPUT_FILES)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBFABRIC_TRANSPORT_SONAME) -fpic -Xlinker --version-script=nvshmem_transport.sym -I$(TRANSPORT_INCDIR) -I$(INCDIR) -Isrc/include $< -o $@ $(LIBFABRIC_TRANSPORT_OUTPUT_FILES) -L$(LIBFABRIC_LIBDIR) -lfabric -L$(CUDA_LIB) -lcudart_static
	ln -sf $(LIBFABRIC_TRANSPORT_REALNAME) $(LIBDIR)/$(LIBFABRIC_TRANSPORT_SONAME)
	ln -sf $(LIBFABRIC_TRANSPORT_SONAME) $(LIBDIR)/$(LIBFABRIC_TRANSPORT)
endif

$(BINDIR)/nvshmem-info: src/util/nvshmem-info.cpp $(LIBDIR)/$(LIBTARGET)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVCUFLAGS) -Isrc/include -I$(INCDIR) $< -o $@ $(LDFLAGS) -L$(LIBDIR) -lnvshmem

$(INCDIR)/%.h : src/include/%.h
	@mkdir -p `dirname $@`
	cp -f $< $@

$(INCDIR)/%.cuh : src/include/%.cuh
	@mkdir -p `dirname $@`
	cp -f $< $@

$(INCDIR)/nvshmem_version.h :
	@mkdir -p $(INCDIR)
	@echo "#ifndef NVSHMEM_VERSION_H" > $@
	@echo "#define NVSHMEM_VERSION_H" >> $@
	@echo "#define NVSHMEM_VENDOR_MAJOR_VERSION $(NVSHMEM_MAJOR)" >> $@
	@echo "#define NVSHMEM_VENDOR_MINOR_VERSION $(NVSHMEM_MINOR)" >> $@
	@echo "#define NVSHMEM_VENDOR_PATCH_VERSION $(NVSHMEM_PATCH)" >> $@
	@echo "#define NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION $(TRANSPORT_VERSION_MAJOR)" >> $@
	@echo "#define NVSHMEM_TRANSPORT_PLUGIN_MINOR_VERSION $(TRANSPORT_VERSION_MINOR)" >> $@
	@echo "#define NVSHMEM_TRANSPORT_PLUGIN_PATCH_VERSION $(TRANSPORT_VERSION_PATCH)" >> $@
	@echo "#define NVSHMEM_BOOTSTRAP_PLUGIN_MAJOR_VERSION $(BOOTSTRAP_VERSION_MAJOR)" >> $@
	@echo "#define NVSHMEM_BOOTSTRAP_PLUGIN_MINOR_VERSION $(BOOTSTRAP_VERSION_MINOR)" >> $@
	@echo "#define NVSHMEM_BOOTSTRAP_PLUGIN_PATCH_VERSION $(BOOTSTRAP_VERSION_PATCH)" >> $@
	@echo "#define NVSHMEM_BUILD_VARS \"$(INFO_BUILD_VARS)\"" >> $@
	@echo "#endif /* NVSHMEM_VERSION_H */" >> $@

.PHONY: $(INCDIR)/nvshmem_build_options.h
$(INCDIR)/nvshmem_build_options.h: Makefile
	@mkdir -p $(INCDIR)
	@echo "#ifndef NVSHMEM_BUILD_OPTIONS_H" > $@
	@echo "#define NVSHMEM_BUILD_OPTIONS_H" >> $@
	@printf  $(BUILD_OPTIONS_STR) >> $@
	@echo "#endif /* NVSHMEM_BUILD_OPTIONS_H */" >> $@

$(INCDIR)/device/pt-to-pt/transfer_device.cuh: src/include/device/pt-to-pt/transfer_device.cuh.in
	@mkdir -p `dirname $@`
	@cp $^ $@

src/comm/device/transfer_device.cu: src/include/device/pt-to-pt/transfer_device.cuh.in
	@cp $^ $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cpp $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(CXX) -c $(LIBINC) $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/%.o : src/%.c $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(CC) -c $(LIBINC) $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cu $(BUILT_HEADERS)
	@mkdir -p `dirname $@`
	$(NVCC) -c $(LIBINC) $(NVCUFLAGS) $(EXTRA_NVCUFLAGS) -rdc=true $< -o $@

.PHONY: clean
clean :
	rm -rf $(NVSHMEM_BUILDDIR)

.PHONY: uninstall
uninstall:
	rm -rf $(NVSHMEM_PREFIX)

.PHONY: purge
purge: clean uninstall

.PHONY: install
install : lib
	mkdir -p $(NVSHMEM_PREFIX)/lib
	mkdir -p $(NVSHMEM_PREFIX)/include
	mkdir -p $(NVSHMEM_PREFIX)/share
	mkdir -p $(NVSHMEM_PREFIX)/bin
	cp -v $(NVSHMEM_BUILDDIR)/bin/* $(NVSHMEM_PREFIX)/bin/
	cp -v $(NVSHMEM_BUILDDIR)/lib/* $(NVSHMEM_PREFIX)/lib/
	cp -P -R -v $(NVSHMEM_BUILDDIR)/include/* $(NVSHMEM_PREFIX)/include/
	cp -P -R -v $(NVSHMEM_BUILDDIR)/share/* $(NVSHMEM_PREFIX)/share/
