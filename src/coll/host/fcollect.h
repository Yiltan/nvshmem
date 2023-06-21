/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_FCOLLECT_CPU_H
#define NVSHMEMI_FCOLLECT_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

template <typename TYPE>
void nvshmemi_call_fcollect_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                             size_t nelems, cudaStream_t stream);

template <typename TYPE>
int nvshmemi_fcollect_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems,
                                cudaStream_t stream) {
#ifdef NVSHMEM_USE_NCCL
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    if (nvshmemi_use_nccl && nvshmemi_get_nccl_dt<TYPE>() != ncclNumTypes) {
        NCCL_CHECK(nccl_ftable.AllGather(source, dest, nelems, nvshmemi_get_nccl_dt<TYPE>(),
                                         (ncclComm_t)teami->nccl_comm, stream));
    } else
#endif
    {
        nvshmemi_call_fcollect_on_stream_kernel<TYPE>(team, dest, source, nelems, stream);
    }
    return 0;
}
#endif /* NVSHMEMI_FCOLLECT_CPU_H */
