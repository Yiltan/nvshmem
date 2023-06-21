/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_DEVICE_COLL_UTILS_H_
#define _NVSHMEMI_DEVICE_COLL_UTILS_H_

#include <type_traits>
#include "cuda.h"
//#include "nvshmem_internal.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "device/pt-to-pt/transfer_device.cuh"
#else
#include "device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "device/team/team_device.cuh"

using namespace std;

#define GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dest, src, nelems, myIdx, groupSize) \
    do {                                                                                      \
        int i;                                                                                \
        for (i = myIdx; i < nelems; i += groupSize) {                                         \
            *((TYPE *)dest + i) = *((TYPE *)src + i);                                         \
        }                                                                                     \
    } while (0)

#ifdef __CUDA_ARCH__
template <typename T, rdxn_ops_t op>
__device__ inline typename enable_if<is_integral<T>::value, T>::type perform_gpu_rdxn(T op1,
                                                                                      T op2) {
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_AND:
            return op1 & op2;
        case RDXN_OPS_OR:
            return op1 | op2;
        case RDXN_OPS_XOR:
            return op1 ^ op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <typename T, rdxn_ops_t op>
__device__ inline typename enable_if<!is_integral<T>::value, T>::type perform_gpu_rdxn(T op1,
                                                                                       T op2) {
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <>
__device__ inline double2 perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(double2 op1, double2 op2) {
    return (op1.x > op2.x) ? op1 : op2;
}

/* This is signaling function used in barrier algorithm.
nvshmem_<type>_signal function cannot be used in barrier because it uses a
combination of P2P path and IB path depending on how the peer GPU is
connected. In contrast to that, this fuction uses either P2P path (when all GPUs
are NVLink connected) or IB path (when any of the GPU is not NVLink connected).

Using this function in barrier is necessary to ensure any previous RMA
operations are visible. When combination of P2P and IB path are used
as in nvshmem_<type>_signal function, it can lead to race conditions.
For example NVLink writes (of data and signal) can overtake IB writes.
And hence the data may not be visible after the barrier operation.
*/
template <typename T>
__device__ inline void nvshmemi_signal_for_barrier(T *dest, const T value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);
    if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        volatile T *dest_actual =
            (volatile T *)((char *)(peer_base_addr) +
                           ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
    } else {
        nvshmemi_transfer_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
    }
}
#endif /* __CUDACC__ */

#endif /* NVSHMEMI_DEVICE_COLL_UTILS_H */
