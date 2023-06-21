/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

void nvshmemi_call_proxy_quiet_entrypoint(cudaStream_t cstrm);
#ifdef __cplusplus
extern "C" {
#endif
void nvshmemx_quiet_on_stream(cudaStream_t cstrm);
#ifdef __cplusplus
}
#endif

void nvshmem_quiet(void) {
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
    NVSHMEMI_CHECK_INIT_STATUS();

    int status = 0;

    int tbitmap = nvshmemi_state->transport_bitmap;
    if (nvshmemi_state->used_internal_streams) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            if (nvshmemi_state->active_internal_streams[s]) {
                cudaStream_t custrm = nvshmemi_state->custreams[s];
                CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
                nvshmemi_state->active_internal_streams[s] = 0;
            }
        }
        nvshmemi_state->used_internal_streams = 0;
    }

    for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr =
                ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
            for (int k = 0; k < nvshmemi_state->npes; k++) {
                if (nvshmemi_state->quiet[j]) {
                    status = nvshmemi_state->quiet[j](tcurr, k, 0);
                }
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "nvshmem_quiet() failed \n");
            }
        }
        tbitmap >>= 1;
    }
out:
    return;
}

void nvshmemi_quiesce_internal_streams(cudaStream_t cstrm) {
    if (nvshmemi_state->used_internal_streams) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            cudaStream_t custrm = nvshmemi_state->custreams[s];
            cudaEvent_t cuev = nvshmemi_state->cuevents[s];

            if (nvshmemi_state->active_internal_streams[s]) {
                CUDA_RUNTIME_CHECK(cudaEventRecord(cuev, custrm));
                CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(cstrm, cuev, 0));
                nvshmemi_state->active_internal_streams[s] = 0;
            }
        }
        nvshmemi_state->used_internal_streams = 0;
    }
}

void nvshmemx_quiet_on_stream(cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(QUIET_ON_STREAM);
    NVSHMEMI_CHECK_INIT_STATUS();

    int tbitmap = nvshmemi_state->transport_bitmap;
    nvshmemi_quiesce_internal_streams(cstrm);

    for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr =
                ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
            if (!tcurr->no_proxy) {
                nvshmemi_call_proxy_quiet_entrypoint(cstrm);
            }
        }
        tbitmap >>= 1;
    }

    return;
}
