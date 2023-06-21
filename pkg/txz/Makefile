# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

#### PACKAGING ####
include ../../common.mk
include ../../version.mk
NVSHMEM_BUILDDIR ?= $(abspath ../../build)
TXZPREPDIR  := $(NVSHMEM_BUILDDIR)/txz
PKGDIR  := $(NVSHMEM_BUILDDIR)/pkg/txz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_ARCH       := $(shell uname -m)

prep: $(TXZTARGETS)
	make -j -C ../.. lic NVSHMEM_MPI_SUPPORT=1 NVSHMEM_SHMEM_SUPPORT=1

build: prep
	make -j -C ../.. NVSHMEM_MPI_SUPPORT=1 NVSHMEM_SHMEM_SUPPORT=1
	@printf "Making hydra package\n"
	mkdir -p $(NVSHMEM_BUILDDIR)/hydra_installation
	../../scripts/install_hydra.sh $(NVSHMEM_BUILDDIR)/hydra_installation $(NVSHMEM_BUILDDIR)
	rm -rf $(NVSHMEM_BUILDDIR)/hydra_installation
	@printf "Building tar.xz package\n"
	(cd $(NVSHMEM_BUILDDIR); bash txz/create_txz.sh)
	mkdir -p $(PKGDIR)
	mv $(NVSHMEM_BUILDDIR)/nvshmem*.txz $(PKGDIR)

src: prep
	@printf "Building tar.xz package\n"
	(cd $(NVSHMEM_BUILDDIR); bash txz/create_src_txz.sh)
	mkdir -p $(PKGDIR)
	mv $(NVSHMEM_BUILDDIR)/nvshmem*.txz $(PKGDIR)

clean:$(TXZTARGETS)
	rm -rf $(NVSHMEM_BUILDDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	cp ../../changelog $(NVSHMEM_BUILDDIR)/
	cp ../../CMakeLists.txt $(NVSHMEM_BUILDDIR)/
	cp ../../common.mk $(NVSHMEM_BUILDDIR)/
	cp ../../version.mk $(NVSHMEM_BUILDDIR)/
	cp ../../Makefile $(NVSHMEM_BUILDDIR)/
	cp ../../NVSHMEMConfig.cmake.in $(NVSHMEM_BUILDDIR)/
	cp ../../nvshmem_transport.sym $(NVSHMEM_BUILDDIR)/
	cp ../../nvshmem_host.sym $(NVSHMEM_BUILDDIR)/
	cp ../../nvshmem_bootstrap.sym $(NVSHMEM_BUILDDIR)/
	mkdir -p $(NVSHMEM_BUILDDIR)/src
	cp -r ../../src/* $(NVSHMEM_BUILDDIR)/src/
	rm -rf $(NVSHMEM_BUILDDIR)/src/pm
	mkdir -p $(NVSHMEM_BUILDDIR)/examples
	cp ../../examples/Makefile $(NVSHMEM_BUILDDIR)/examples/
	cp ../../examples/CMakeLists.txt $(NVSHMEM_BUILDDIR)/examples/
	cp ../../examples/*.c* $(NVSHMEM_BUILDDIR)/examples/
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest
	cp ../../perftest/Makefile $(NVSHMEM_BUILDDIR)/perftest/
	cp ../../perftest/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest/common
	cp ../../perftest/common/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/common/
	cp ../../perftest/common/*.c* $(NVSHMEM_BUILDDIR)/perftest/common/
	cp ../../perftest/common/*.h* $(NVSHMEM_BUILDDIR)/perftest/common/
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest/device/pt-to-pt
	cp ../../perftest/device/pt-to-pt/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/device/pt-to-pt/
	cp ../../perftest/device/pt-to-pt/*.c* $(NVSHMEM_BUILDDIR)/perftest/device/pt-to-pt/
	cp ../../perftest/device/pt-to-pt/*.h* $(NVSHMEM_BUILDDIR)/perftest/device/pt-to-pt/ 2>/dev/null || :
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest/device/coll
	cp ../../perftest/device/coll/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/device/coll/
	cp ../../perftest/device/coll/*.c* $(NVSHMEM_BUILDDIR)/perftest/device/coll/
	cp ../../perftest/device/coll/*.h* $(NVSHMEM_BUILDDIR)/perftest/device/coll/ 2>/dev/null || :
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest/host/pt-to-pt
	cp ../../perftest/host/pt-to-pt/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/host/pt-to-pt/
	cp ../../perftest/host/pt-to-pt/*.c* $(NVSHMEM_BUILDDIR)/perftest/host/pt-to-pt/
	cp ../../perftest/host/pt-to-pt/*.h* $(NVSHMEM_BUILDDIR)/perftest/host/pt-to-pt/ 2>/dev/null || :
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest/host/coll
	cp ../../perftest/host/coll/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/host/coll/
	cp ../../perftest/host/coll/*.c* $(NVSHMEM_BUILDDIR)/perftest/host/coll/
	cp ../../perftest/host/coll/*.h* $(NVSHMEM_BUILDDIR)/perftest/host/coll/ 2>/dev/null || :
	mkdir -p $(NVSHMEM_BUILDDIR)/perftest/host/init
	cp ../../perftest/host/init/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/host/init/
	cp ../../perftest/host/init/*.c* $(NVSHMEM_BUILDDIR)/perftest/host/init/
	cp ../../perftest/host/init/*.h* $(NVSHMEM_BUILDDIR)/perftest/host/init/ 2>/dev/null || :
	cp ../../perftest/device/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/device
	cp ../../perftest/host/CMakeLists.txt $(NVSHMEM_BUILDDIR)/perftest/host
	mkdir -p $(NVSHMEM_BUILDDIR)/pkg
	cp -rL ../../pkg/* $(NVSHMEM_BUILDDIR)/pkg
	rm -f $(NVSHMEM_BUILDDIR)/pkg/txz/create_src_txz.sh.in
	mkdir -p $(NVSHMEM_BUILDDIR)/scripts
	cp -r ../../scripts/install_hydra.sh $(NVSHMEM_BUILDDIR)/scripts/
	cp -r ../../scripts/test_cxx11.sh $(NVSHMEM_BUILDDIR)/scripts/
	sed -e "s/\$${nvshmem:Major}/$(NVSHMEM_MAJOR)/g" \
	    -e "s/\$${nvshmem:Minor}/$(NVSHMEM_MINOR)/g" \
	    -e "s/\$${nvshmem:Patch}/$(NVSHMEM_PATCH)/g" \
		-e "s/\$${nvshmem:Tests}/$(NVSHMEM_PACKAGE_INCLUDE_TESTS)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
		-e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@
