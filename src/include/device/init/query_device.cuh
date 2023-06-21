/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef QUERY_DEVICE_CUH
#define QUERY_DEVICE_CUH
//#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__
__device__ inline int nvshmem_my_pe(void) { return nvshmemi_device_state_d.mype; }

__device__ inline int nvshmem_n_pes(void) { return nvshmemi_device_state_d.npes; }

__device__ inline void nvshmem_info_get_name(char *name) {
    size_t i;
    const char *str = NVSHMEM_VENDOR_STRING;

    /* Copy up to NVSHMEM_MAX_NAME_LEN-1 chars, then add NULL terminator */
    for (i = 0; i < NVSHMEM_MAX_NAME_LEN - 1 && str[i] != '\0'; i++) name[i] = str[i];

    name[i] = '\0';
}

__device__ inline void nvshmem_info_get_version(int *major, int *minor) {
    *major = NVSHMEM_MAJOR_VERSION;
    *minor = NVSHMEM_MINOR_VERSION;
}

__device__ inline int nvshmemx_my_pe(nvshmemx_team_t team) {
    return nvshmem_team_my_pe((nvshmem_team_t)team);
}

__device__ inline int nvshmemx_n_pes(nvshmemx_team_t team) {
    return nvshmem_team_n_pes((nvshmem_team_t)team);
}
#endif
#endif
