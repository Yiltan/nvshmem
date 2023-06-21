/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef FCOLLECT_DEVICE_CUH
#define FCOLLECT_DEVICE_CUH
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "device/pt-to-pt/transfer_device.cuh"
#else
#include "device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "utils.cuh"

#ifdef __CUDA_ARCH__
template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_allpush_ll_threadgroup(nvshmem_team_t team, T *dest,
                                                                const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const size_t fcollect_ll_threshold = nvshmemi_device_state_d.fcollect_ll_threshold;
    const size_t fcollect_count = teami->fcollect_count;
    const uint32_t ll_flag = teami->fcollect_count;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, FCOLLECT) +
                 (2 * teami->size * fcollect_ll_threshold * (fcollect_count % 2));
    const int my_pe_in_team = nvshmem_team_my_pe(team);
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    const size_t pack_offset = my_pe_in_team * nelems * (sizeof(T) / sizeof(uint32_t));
    int next_rank, prev_pe_in_team;
    size_t prev_offset;
    int ii_start, ii_inc;
    void *peer_addr;

    nvshmemi_packLL<T, SCOPE>((uint64_t *)pWrk + pack_offset, source, nelems, ll_flag);
    if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        int warp_id, thread_id;
        thread_id = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        warp_id = myIdx / 32;
        ii_start = warp_id;
        ii_inc = (groupSize + 31) / 32;
        for (int ii = ii_start + 1; ii < teami->size; ii += ii_inc) {
            next_rank = teami->start + ((my_pe_in_team + ii) % teami->size) * teami->stride;
            if ((peer_addr = nvshmem_ptr(pWrk, next_rank)) != NULL) {
                for (int j = 2 * thread_id; j < (nelems * sizeof(T)) / sizeof(uint32_t); j += 64) {
                    uint32_t val1 = *((uint32_t *)source + j);
                    uint32_t val2 = *((uint32_t *)source + j + 1);
                    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(
                                     (uint64_t *)peer_addr + pack_offset + j),
                                 "r"(val1), "r"(ll_flag), "r"(val2), "r"(ll_flag));
                }
            } else {
                nvshmemi_put_nbi_threadgroup<uint64_t, NVSHMEMI_THREADGROUP_WARP>(
                    (uint64_t *)pWrk + pack_offset, (uint64_t *)pWrk + pack_offset,
                    nelems * sizeof(T) / sizeof(uint32_t), next_rank);
            }
        }
        for (int ii = ii_start; ii < teami->size; ii += ii_inc) {
            prev_pe_in_team = (my_pe_in_team - ii + teami->size) % teami->size;
            prev_offset = nelems * prev_pe_in_team * (sizeof(T) / sizeof(uint32_t));
            nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                dest + (prev_pe_in_team * nelems), (uint64_t *)pWrk + prev_offset, nelems, ll_flag);
        }
    } else {
        for (int ii = 1; ii < teami->size; ii += 1) {
            next_rank = teami->start + ((my_pe_in_team + ii) % teami->size) * teami->stride;
            nvshmemi_put_nbi_threadgroup<uint64_t, SCOPE>(
                (uint64_t *)pWrk + pack_offset, (uint64_t *)pWrk + pack_offset,
                nelems * sizeof(T) / sizeof(uint32_t), next_rank);
        }
        for (int ii = 0; ii < teami->size; ii += 1) {
            prev_pe_in_team = (my_pe_in_team - ii + teami->size) % teami->size;
            prev_offset = nelems * prev_pe_in_team * (sizeof(T) / sizeof(uint32_t));
            nvshmemi_recvLL<T, SCOPE>(dest + (prev_pe_in_team * nelems),
                                      (uint64_t *)pWrk + prev_offset, nelems, ll_flag);
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                             const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int next_rank;
    int next_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    // nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        next_offset = nelems * ((mype - PE_start) / stride);
        nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest + next_offset, source, nelems, next_rank);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_p2p_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int next_rank;
    int next_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    T *dst_ptr;
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        next_offset = nelems * my_idx_in_active_set;
        dst_ptr = (T *)nvshmem_ptr((void *)(dest + next_offset), next_rank);
        nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_threadgroup(nvshmem_team_t team, T *dest, const T *source,
                                                     size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if (!myIdx) /* Only one thread should increment fcollect_count */
        nvshmemi_device_state_d.team_pool[team]->fcollect_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
    if (sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0 &&
        nvshmemi_device_state_d.fcollect_ll_threshold >= (nelems * sizeof(T)) &&
        SCOPE == NVSHMEMI_THREADGROUP_BLOCK)
        nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE>(team, dest, source, nelems);
    else if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS)
        nvshmemi_fcollect_p2p_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
    else {
        nvshmemi_fcollect_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
    }
}

#define DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)      \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                   \
        nvshmem##SC_PREFIX##_##TYPENAME##_fcollect##SC_SUFFIX(nvshmem_team_t team, TYPE *dest,     \
                                                              const TYPE *source, size_t nelems) { \
        nvshmemi_fcollect_threadgroup<TYPE, nvshmemi_threadgroup_##SC>(team, dest, source,         \
                                                                       nelems);                    \
        return 0;                                                                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP,
                                                 thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP, block,
                                                 _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP

#endif /* __CUDA_ARCH__ */
#endif /* FCOLLECT_DEVICE_CUH */
