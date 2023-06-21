/****
 * Copyright (c) 2016-2018, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

#include <iostream>
#include <cassert>
#include "util.h"

int nvshmemi_setup_collective_launch(nvshmemi_state_t *state) {
    int leastPriority, greatestPriority, status = 0;
    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&(state->cu_dev_attrib.multi_processor_count),
                               cudaDevAttrMultiProcessorCount, state->device_id),
        status, out);
    state->cu_dev_attrib.cooperative_launch = 0;

    CUDA_RUNTIME_CHECK_GOTO(cudaDeviceGetAttribute(&(state->cu_dev_attrib.cooperative_launch),
                                                   cudaDevAttrCooperativeLaunch, state->device_id),
                            status, out);

    if (!state->cu_dev_attrib.cooperative_launch)
        NVSHMEMI_WARN_PRINT(
            "Cooperative launch not supported on PE %d; GPU-side synchronize may cause hang\n",
            state->mype);

    CUDA_RUNTIME_CHECK_GOTO(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaStreamCreateWithPriority(&state->claunch_params.stream,
                                                         cudaStreamNonBlocking, greatestPriority),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaEventCreate(&state->claunch_params.begin_event, cudaEventDisableTiming), status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaEventCreate(&state->claunch_params.end_event, cudaEventDisableTiming), status, out);

out:
    return status;
}

int nvshmemi_teardown_collective_launch(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_teardown_collective_launch");
    int status = 0;

    if (!nvshmemi_is_nvshmem_initialized) goto out;

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamDestroy(state->claunch_params.stream), status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(state->claunch_params.begin_event), status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(state->claunch_params.end_event), status, out);

out:
    return status;
}
