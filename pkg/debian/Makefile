#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT.txt for license information
#

include ../../common.mk
include ../../version.mk
BUILDDIR ?= $(abspath ../../build)
DEBPREPDIR := $(BUILDDIR)/debian
PKGDIR  := $(BUILDDIR)/pkg/deb/

DEBGEN_IN  := $(wildcard *.in)
DEBGEN     := $(DEBGEN_IN:.in=)
DEBFILES   := changelog compat copyright libnvshmem.install rules $(DEBGEN)
DEBTARGETS := $(patsubst %, $(DEBPREPDIR)/%, $(DEBFILES))

PKG_TIMESTAMP  := $(shell date -R)
PKG_ARCH       ?= $(shell dpkg-architecture -qDEB_HOST_ARCH)
PKG_MULTIARCH  ?= $(shell dpkg-architecture -qDEB_HOST_MULTIARCH)

build : $(DEBTARGETS)
	$(MAKE) -C ../txz build BUILDDIR=$(BUILDDIR)
	rm -f $(BUILDDIR)/Makefile
	rm -f $(BUILDDIR)/CMakeLists.txt
	@printf "Building Debian package\n"
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../libnvshmem*.deb $(PKGDIR)/

clean:
	rm -Rf $(DEBPREPDIR) $(PKGDIR)

$(DEBPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	sed -e "s/\$${nvshmem:Major}/$(NVSHMEM_MAJOR)/g" \
	    -e "s/\$${nvshmem:Minor}/$(NVSHMEM_MINOR)/g" \
	    -e "s/\$${nvshmem:Patch}/$(NVSHMEM_PATCH)/g" \
	    -e "s/\$${nvshmem:Suffix}/$(NVSHMEM_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Timestamp}/$(PKG_TIMESTAMP)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    -e "s/\$${pkg:MultiArch}/$(PKG_MULTIARCH)/g" \
	    $< > $@

$(DEBPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	cp -f $< $@
