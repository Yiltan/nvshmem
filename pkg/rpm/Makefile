#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information
#

include ../../common.mk
include ../../version.mk
BUILDDIR ?= $(abspath ../../build)
RPMPREPDIR := $(BUILDDIR)/redhat
PKGDIR  := $(BUILDDIR)/pkg/rpm/

RPMGEN_IN  := $(wildcard *.in)
RPMGEN     := $(RPMGEN_IN:.in=)
RPMFILES   := $(RPMGEN)
RPMTARGETS := $(patsubst %, $(RPMPREPDIR)/%, $(RPMFILES))

PKG_TIMESTAMP  := $(shell date -R)
ARCH           := $(shell uname -m)
PKG_ARCH       ?= $(shell uname -m)
PKG_MULTIARCH  ?= $(shell $(CXX) -print-multiarch)
ifeq ($(PKG_MULTIARCH),)
# Hardwire the PKG_MULTIARCH directory as the RHEL6 distribution agnostic compiler (gcc 4.8.3) doesn't set it
PKG_MULTIARCH  := $(ARCH)-linux-gnu
endif

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
define NVSHMEM_PMIX_PKG_INSTALL
install -m 644 lib/nvshmem_bootstrap_pmix.so.${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_bootstrap_pmix.so.${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_pmix.so.${BOOTSTRAP_VERSION_MAJOR}\nln -s nvshmem_bootstrap_pmix.so.${BOOTSTRAP_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_pmix.so
endef
else
NVSHMEM_PMIX_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
define NVSHMEM_MPI_PKG_INSTALL
install -m 644 lib/nvshmem_bootstrap_mpi.so.${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_bootstrap_mpi.so.${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_mpi.so.${BOOTSTRAP_VERSION_MAJOR}\nln -s nvshmem_bootstrap_mpi.so.${BOOTSTRAP_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_mpi.so
endef
else
NVSHMEM_MPI_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
define NVSHMEM_SHMEM_PKG_INSTALL
install -m 644 lib/nvshmem_bootstrap_shmem.so.${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_bootstrap_shmem.so.${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_shmem.so.${BOOTSTRAP_VERSION_MAJOR}\nln -s nvshmem_bootstrap_shmem.so.${BOOTSTRAP_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_bootstrap_shmem.so
endef
else
NVSHMEM_SHMEM_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_UCX_SUPPORT), 1)
define NVSHMEM_UCX_PKG_INSTALL
install -m 644 lib/nvshmem_transport_ucx.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_transport_ucx.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ucx.so.${TRANSPORT_VERSION_MAJOR}\nln -s nvshmem_transport_ucx.so.${TRANSPORT_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ucx.so
endef
else
NVSHMEM_UCX_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_IBRC_SUPPORT), 1)
define NVSHMEM_IBRC_PKG_INSTALL
install -m 644 lib/nvshmem_transport_ibrc.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_transport_ibrc.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ibrc.so.${TRANSPORT_VERSION_MAJOR}\nln -s nvshmem_transport_ibrc.so.${TRANSPORT_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ibrc.so
endef
else
NVSHMEM_IBRC_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_IBDEVX_SUPPORT), 1)
define NVSHMEM_IBDEVX_PKG_INSTALL
install -m 644 lib/nvshmem_transport_ibdevx.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_transport_ibdevx.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ibdevx.so.${TRANSPORT_VERSION_MAJOR}\nln -s nvshmem_transport_ibdevx.so.${TRANSPORT_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ibdevx.so
endef
else
NVSHMEM_IBDEVX_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_IBGDA_SUPPORT), 1)
define NVSHMEM_IBGDA_PKG_INSTALL
install -m 644 lib/nvshmem_transport_ibgda.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_transport_ibgda.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ibgda.so.${TRANSPORT_VERSION_MAJOR}\nln -s nvshmem_transport_ibgda.so.${TRANSPORT_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_ibgda.so
endef
else
NVSHMEM_IBGDA_PKG_INSTALL=""
endif

ifeq ($(NVSHMEM_LIBFABRIC_SUPPORT), 1)
define NVSHMEM_LIBFABRIC_PKG_INSTALL
install -m 644 lib/nvshmem_transport_libfabric.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}\nln -s nvshmem_transport_libfabric.so.${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_libfabric.so.${TRANSPORT_VERSION_MAJOR}\nln -s nvshmem_transport_libfabric.so.${TRANSPORT_VERSION_MAJOR} \$${RPM_BUILD_ROOT}/%{_libdir}/nvshmem_transport_libfabric.so
endef
else
NVSHMEM_LIBFABRIC_PKG_INSTALL=""
endif

build: $(RPMTARGETS)
	$(MAKE) -C ../txz build BUILDDIR=$(BUILDDIR)
	@printf "Building Redhat package\n"
	mkdir -p $(PKGDIR)
	rpmbuild --define "_sourcedir $(BUILDDIR)/pkg/txz" \
                 --define "_rpmdir $(PKGDIR)" \
                 --define "_builddir $(PKGDIR)/build/" \
                 --define "_buildrootdir $(PKGDIR)/buildroot/" \
                 -bb $(BUILDDIR)/redhat/nvshmem.spec

clean:
	rm -Rf $(RPMPREPDIR) $(PKGDIR)

$(RPMPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	sed -e "s/\$${nvshmem:Major}/$(NVSHMEM_MAJOR)/g" \
	    -e "s/\$${nvshmem:Minor}/$(NVSHMEM_MINOR)/g" \
	    -e "s/\$${nvshmem:Patch}/$(NVSHMEM_PATCH)/g" \
	    -e "s/\$${bootstrap:Major}/$(BOOTSTRAP_VERSION_MAJOR)/g" \
	    -e "s/\$${bootstrap:Minor}/$(BOOTSTRAP_VERSION_MINOR)/g" \
	    -e "s/\$${bootstrap:Patch}/$(BOOTSTRAP_VERSION_PATCH)/g" \
	    -e "s/\$${nvshmem:Suffix}/$(NVSHMEM_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Timestamp}/$(PKG_TIMESTAMP)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    -e "s/\$${pkg:MultiArch}/$(PKG_MULTIARCH)/g" \
	    -e "s,\$${nvshmem:pmix_install},$(NVSHMEM_PMIX_PKG_INSTALL),g" \
	    -e "s,\$${nvshmem:mpi_install},$(NVSHMEM_MPI_PKG_INSTALL),g" \
	    -e "s,\$${nvshmem:shmem_install},$(NVSHMEM_SHMEM_PKG_INSTALL),g" \
		-e "s,\$${nvshmem:ucx_install},$(NVSHMEM_UCX_PKG_INSTALL),g" \
		-e "s,\$${nvshmem:ibrc_install},$(NVSHMEM_IBRC_PKG_INSTALL),g" \
		-e "s,\$${nvshmem:ibdevx_install},$(NVSHMEM_IBDEVX_PKG_INSTALL),g" \
		-e "s,\$${nvshmem:ibgda_install},$(NVSHMEM_IBGDA_PKG_INSTALL),g" \
		-e "s,\$${nvshmem:libfabric_install},$(NVSHMEM_LIBFABRIC_PKG_INSTALL),g" \
	    $< > $@

$(RPMPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	cp -f $< $@
