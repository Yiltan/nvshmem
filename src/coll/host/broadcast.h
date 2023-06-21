/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BCAST_COMMON_CPU_H
#define NVSHMEMI_BCAST_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
void nvshmemi_call_broadcast_on_stream_kernel(nvshmem_team_t team, T *dest, const T *source,
                                              size_t nelems, int PE_root, cudaStream_t stream);

template <typename TYPE>
int nvshmemi_broadcast_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems,
                                 int PE_root, cudaStream_t stream) {
#ifdef NVSHMEM_USE_NCCL
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    if (nvshmemi_use_nccl && nvshmemi_get_nccl_dt<TYPE>() != ncclNumTypes) {
        NCCL_CHECK(nccl_ftable.Broadcast(source, dest, nelems, nvshmemi_get_nccl_dt<TYPE>(),
                                         PE_root, (ncclComm_t)teami->nccl_comm, stream));
    } else
#endif
    {
        nvshmemi_call_broadcast_on_stream_kernel<TYPE>(team, dest, source, nelems, PE_root, stream);
    }
    return 0;
}
#endif
