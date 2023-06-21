/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"
#include "nvshmemx_error.h"

void nvshmem_fence(void) {
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
    NVSHMEMI_CHECK_INIT_STATUS();

    int status;
    int tbitmap = nvshmemi_state->transport_bitmap;
    for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr =
                ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
            if ((tcurr->attr & NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS)) {
                for (int s = 0; s < MAX_PEER_STREAMS; s++) {
                    cudaStream_t custrm = nvshmemi_state->custreams[s];
                    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
                }
            } else if (nvshmemi_state->fence[j]) {
                for (int k = 0; k < nvshmemi_state->npes; k++) {
                    status = nvshmemi_state->fence[j](tcurr, k, 0);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "nvshmem_fence() failed \n");
                }
            }
        }
        tbitmap >>= 1;
    }
out:
    return;
}
