/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include <string>
#include <typeinfo>

using namespace std;

template <typename TYPE>
__global__ void alltoall_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                          size_t nelems) {
#ifdef __CUDA_ARCH__
    if (!blockIdx.x)
        nvshmemi_alltoall_threadgroup<TYPE, NVSHMEMI_THREADGROUP_BLOCK>(team, dest, source, nelems);
#endif
}

template <typename TYPE>
void nvshmemi_call_alltoall_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                             size_t nelems, cudaStream_t stream) {
    int tmp;
    string type_str(typeid(TYPE).name());
    if (nvshmemi_alltoall_maxblocksize.find(type_str) == nvshmemi_alltoall_maxblocksize.end()) {
        CUDA_RUNTIME_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_alltoall_maxblocksize[type_str],
            alltoall_on_stream_kernel<TYPE>));
    }
    int num_threads_per_block = (nvshmemi_alltoall_maxblocksize[type_str] > nelems)
                                    ? nelems
                                    : nvshmemi_alltoall_maxblocksize[type_str];
    int num_blocks = 1;
    alltoall_on_stream_kernel<TYPE>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(team, dest, source, nelems);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

#define INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(TYPE) \
    template void nvshmemi_call_alltoall_on_stream_kernel<TYPE>(  \
        nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t);
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(uint8_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(uint16_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(uint32_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(uint64_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(int8_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(int16_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(int32_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(int64_t)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(float)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(char)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(double)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(long long)
INSTANTIATE_NVSHMEMI_CALL_ALLTOALL_ON_STREAM_KERNEL(unsigned long long)
