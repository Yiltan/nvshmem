/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef NVSHMEM_BOOTSTRAP_H
#define NVSHMEM_BOOTSTRAP_H

#include "nvshmem_bootstrap_defines.h"
#include "nvshmem_version.h"
/* Version = major * 10000 + minor * 100 + patch*/
/* ABI Introduced in NVSHMEM 2.8.0 */
#define NVSHMEMI_BOOTSTRAP_ABI_VERSION                \
    (NVSHMEM_BOOTSTRAP_PLUGIN_MAJOR_VERSION * 10000 + \
     NVSHMEM_BOOTSTRAP_PLUGIN_MINOR_VERSION * 100 + NVSHMEM_BOOTSTRAP_PLUGIN_PATCH_VERSION)

static bool nvshmemi_is_bootstrap_compatible(int bootstrap_version, int nvshmem_version) {
    if (bootstrap_version == nvshmem_version)
        return true;
    else
        return false;
}

#if __cplusplus
extern "C" {
#endif
int nvshmemi_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle,
                                   const int nvshmem_version);
#if __cplusplus
}
#endif

#endif
