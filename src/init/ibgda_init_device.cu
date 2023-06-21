/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "nvshmemi_ibgda.h"
#include "nvshmemx_error.h"
#include "util.h"

__constant__ nvshmemi_gic_device_state_t nvshmemi_gic_device_state_d;

int nvshmemi_gic_set_device_state(nvshmemi_gic_device_state_t *gic_device_state) {
    int status = cudaMemcpyToSymbol(nvshmemi_gic_device_state_d, (void *)gic_device_state,
                                    sizeof(nvshmemi_gic_device_state_t), 0, cudaMemcpyHostToDevice);
    return status;
}

/**
 * Variables in constant memory may be updated after nvshmem_init has already been called.
 * The same variable in constant memory has different addresses in libnvshmem_host.so and
 * libnvshmem_device.a. Thus, host API that causes GIC states to change must also call this API.
 */
int nvshmemi_gic_update_device_state() {
    int status = 0;
    nvshmemi_gic_device_state_t *gic_device_state;
    nvshmemx_gic_get_device_state((void **)&gic_device_state);
    status = nvshmemi_gic_set_device_state(gic_device_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemi_gic_set_device_state failed \n");

out:
    return status;
}
