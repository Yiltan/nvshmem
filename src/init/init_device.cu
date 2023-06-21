/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <stdio.h>
#include "nvshmemx_error.h"
#include "util.h"
#include <algorithm>
#include "device/pt-to-pt/proxy_device.cuh"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "nvshmemi_ibgda.h"
#endif

__constant__ nvshmemi_device_state_t nvshmemi_device_state_d;
bool nvshmemi_is_device_state_set = false;
const nvshmemi_version_t nvshmemi_device_lib_version = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
__constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};

#ifdef __CUDA_ARCH__
__device__ void nvshmem_global_exit(int status) {
    if (nvshmemi_device_state_d.proxy > NVSHMEMI_PROXY_NONE) {
        nvshmemi_proxy_global_exit(status);
    } else {
        /* TODO: Add device side printing macros */
        printf(
            "Device side proxy was called, but is not supported under your configuration. "
            "Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set it to false.\n");
        assert(0);
    }
}
#endif

int nvshmemi_set_device_state(nvshmemi_device_state_t *nvshmemi_device_state) {
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_device_state_d, (void *)nvshmemi_device_state,
                                          sizeof(nvshmemi_device_state_t), 0,
                                          cudaMemcpyHostToDevice));
    nvshmemi_is_device_state_set = 1;
    return 0;
}

void nvshmemi_check_state_and_init() {
    if (!nvshmemi_is_device_state_set) {
        if (!nvshmemi_is_nvshmem_bootstrapped)
            NVSHMEMI_ERROR_EXIT("nvshmem API called before nvshmem_init \n");
        if (!nvshmemi_is_nvshmem_initialized) {
            if (nvshmemx_internal_common_init()) {
                NVSHMEMI_ERROR_EXIT("nvshmem initialization failed, exiting \n");
            }
        }
        nvshmemi_device_state_t *nvshmemi_device_state;
        nvshmemx_get_device_state(&nvshmemi_device_state);
        nvshmemi_set_device_state(nvshmemi_device_state);

#ifdef NVSHMEM_IBGDA_SUPPORT
        nvshmemi_gic_update_device_state();
#endif
    }
}

static int handle_state_change() {
#ifdef NVSHMEM_IBGDA_SUPPORT
    return nvshmemi_gic_update_device_state();
#else
    return 0;
#endif
}

int nvshmemi_init_thread(int requested_thread_support, int *provided_thread_support,
                         unsigned int bootstrap_flags, nvshmemx_init_attr_t *bootstrap_attr,
                         nvshmemi_version_t nvshmem_app_version) {
    int status = 0;
    nvshmemi_device_state_t *nvshmemi_device_state;
    if (nvshmemi_is_version_compatible(nvshmemi_device_lib_version, nvshmem_app_version) != 0) {
        printf(
            "NVSHMEM version used in application does not match with NVSHMEM device library "
            "version\n");
        return 1;
    }

    nvshmemi_check_state_and_init_fn_ptr = &nvshmemi_check_state_and_init;
    nvshmemi_init_counter++;
    status =
        nvshmemx_internal_init_thread(requested_thread_support, provided_thread_support,
                                      bootstrap_flags, bootstrap_attr, nvshmemi_device_lib_version);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem_internal_init_thread failed \n");
    if (nvshmemi_is_nvshmem_initialized) {
        nvshmemx_get_device_state(&nvshmemi_device_state);
        nvshmemi_set_device_state(nvshmemi_device_state);
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    nvshmemi_gic_update_device_state();
#endif
    nvshmemi_register_state_change_handler(handle_state_change);

out:
    return status;
}
