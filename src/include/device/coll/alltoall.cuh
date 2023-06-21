/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef ALLTOALL_DEVICE_CUH
#define ALLTOALL_DEVICE_CUH
#include <cstdio>
#include <cassert>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.cuh"
#include "barrier.cuh"

#ifdef __CUDA_ARCH__

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_alltoall_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                             const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int stride = teami->stride;
    int PE_size = teami->size;
    int next_rank, src_offset, dst_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / stride;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    uint64_t *psync = (uint64_t *)nvshmemi_team_get_psync(teami, ALLTOALL);
    uint64_t *pwrk = &teami->alltoall_pwrk[teami->alltoall_count % 2];
    bool need_fence = false;

    dst_offset = nelems * my_idx_in_active_set;

    /* Do remote ops and local ops < 16 bytes from a single thread */
    /* TODO: Find a more optimal transfer point than 16 bytes */
    for (int i = myIdx; i < PE_size; i += groupSize) {
        next_rank = PE_start + ((my_idx_in_active_set + i) % PE_size) * stride;
        void *peer_base_addr = (void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank);
        src_offset = nelems * ((next_rank - PE_start) / stride);
        if (!peer_base_addr) {
            /* We are breaking rank with the rest of the group here so send the RMA with thread
             * scope. */
            nvshmemi_transfer_put_signal<NVSHMEMI_THREADGROUP_THREAD>(
                (void *)(dest + dst_offset), (void *)(source + src_offset), nelems * sizeof(T),
                (void *)psync, 1ULL, NVSHMEMI_AMO_SIGNAL_ADD, next_rank, true);
            atomicAdd((unsigned long long *)pwrk, 1ULL);
        } else if ((nelems * sizeof(T)) <= 16) {
            nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                dest + dst_offset, source + src_offset, nelems, next_rank);
            need_fence = true;
        }
    }

    /* A fence and signal is required - note that we can skip any size check here because it's
     * inherent in the boolean. */
    if (need_fence) {
        __threadfence_system();
        for (int i = myIdx; i < PE_size; i += groupSize) {
            next_rank = PE_start + ((my_idx_in_active_set + i) % PE_size) * stride;
            void *peer_base_addr = (void *)__ldg(
                (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank);
            if (peer_base_addr) {
                nvshmemi_signal_op(psync, 1ULL, NVSHMEMI_AMO_SIGNAL_ADD, next_rank);
                atomicAdd((unsigned long long *)pwrk, 1ULL);
            }
        }
    }

    if ((nelems * sizeof(T)) > 16) {
        for (int ii = 0; ii < PE_size; ii++) {
            next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
            src_offset = nelems * ((next_rank - PE_start) / stride);
            void *peer_base_addr = (void *)__ldg(
                (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank);
            if (peer_base_addr) {
                need_fence = true;
                nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest + dst_offset, source + src_offset,
                                                       nelems, next_rank);
            }
        }
        if (need_fence) {
            __threadfence_system();
            for (int ii = 0; ii < PE_size; ii++) {
                if (myIdx == 0) {
                    next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
                    void *peer_base_addr = (void *)__ldg(
                        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                        next_rank);
                    if (peer_base_addr) {
                        nvshmemi_signal_op(psync, 1ULL, NVSHMEMI_AMO_SIGNAL_ADD, next_rank);
                        atomicAdd((unsigned long long *)pwrk, 1ULL);
                    }
                }
            }
        }
    }

    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemi_transfer_quiet<SCOPE>(false);
    if (myIdx == 0) {
        nvshmemi_wait_until_greater_than_equals<uint64_t>(psync, *pwrk,
                                                          NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GE);
        teami->alltoall_count++;
        nvshmemi_transfer_enforce_consistency_at_target(false);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_alltoall_p2p_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int stride = teami->stride;
    int PE_size = teami->size;
    int next_rank;
    int src_offset;
    int dst_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    T *dst_ptr;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        src_offset = nelems * ((next_rank - PE_start) / stride);
        dst_offset = nelems * ((mype - PE_start) / stride);
        dst_ptr = (T *)nvshmem_ptr((void *)(dest + dst_offset), next_rank);
        nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source + src_offset, nelems * sizeof(T));
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_alltoall_threadgroup(nvshmem_team_t team, T *dest, const T *source,
                                                     size_t nelems) {
    if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS)
        nvshmemi_alltoall_p2p_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
    else
        nvshmemi_alltoall_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
}

#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)      \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                   \
        nvshmem##SC_PREFIX##_##TYPENAME##_alltoall##SC_SUFFIX(nvshmem_team_t team, TYPE *dest,     \
                                                              const TYPE *source, size_t nelems) { \
        nvshmemi_alltoall_threadgroup<TYPE, nvshmemi_threadgroup_##SC>(team, dest, source,         \
                                                                       nelems);                    \
        return 0;                                                                                  \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP,
                                                 thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, block,
                                                 _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_alltoallmem(nvshmem_team_t team, void *dest,
                                                                 const void *source,
                                                                 size_t nelems) {
    nvshmemi_alltoall_threadgroup<char, nvshmemi_threadgroup_thread>(team, (char *)dest,
                                                                     (const char *)source, nelems);
    return 0;
}

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmemx_alltoallmem_warp(nvshmem_team_t team,
                                                                       void *dest,
                                                                       const void *source,
                                                                       size_t nelems) {
    nvshmemi_alltoall_threadgroup<char, nvshmemi_threadgroup_warp>(team, (char *)dest,
                                                                   (const char *)source, nelems);
    return 0;
}

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmemx_alltoallmem_block(nvshmem_team_t team,
                                                                        void *dest,
                                                                        const void *source,
                                                                        size_t nelems) {
    nvshmemi_alltoall_threadgroup<char, nvshmemi_threadgroup_block>(team, (char *)dest,
                                                                    (const char *)source, nelems);
    return 0;
}

#endif /* __CUDA_ARCH__ */
#endif /* ALLTOALL_DEVICE_CUH */
