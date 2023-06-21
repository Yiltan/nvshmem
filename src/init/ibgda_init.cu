/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "nvshmemi_ibgda.h"
#include "nvshmemx_error.h"
#include "util.h"

// Cache CUDA __constant__ variables on host
nvshmemi_gic_device_state_t nvshmemi_gic_device_state;

void nvshmemx_gic_get_device_state(void **gic_device_state) {
    *gic_device_state = (void *)&nvshmemi_gic_device_state;
}
