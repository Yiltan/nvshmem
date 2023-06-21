/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef REDUCE_DEVICE_CUH
#define REDUCE_DEVICE_CUH

#include "device/pt-to-pt/proxy_device.cuh"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "device/pt-to-pt/transfer_device.cuh"
#else
#include "device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "utils.cuh"
#include "nvshmemi_team.h"
#include "fcollect.cuh"
#include "broadcast.cuh"

#ifdef __CUDA_ARCH__

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup(TYPE *x, TYPE *y, TYPE *z,
                                                            size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int i;
    for (i = myIdx; i < nelems; i += groupSize) {
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], y[i]);
    }
}

template <typename TYPE, rdxn_ops_t OP>
__device__ static inline void gpu_rdxn_on_demand_2(int start, int stride, int size, TYPE *dest,
                                                   const TYPE *source, size_t nelems, TYPE *pWrk,
                                                   volatile long *pSync,
                                                   volatile long *sync_counter) {
    int next_rank = -1;
    TYPE *op1 = NULL, *op2 = NULL;
    size_t i;
    volatile TYPE *tmp_operand;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp_operand = (TYPE *)pWrk;
    nvshmemi_put_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems,
                                                                nvshmemi_device_state_d.mype);
    for (i = 1; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>((TYPE *)tmp_operand, source,
                                                                        nelems, next_rank);
        nvshmem_quiet();
        sync_dissem_threadgroup_2<NVSHMEMI_THREADGROUP_THREAD>(start, stride, size, pSync,
                                                               sync_counter);
        op1 = (TYPE *)dest;
        op2 = (TYPE *)tmp_operand;
        gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(op1, op2, op1, nelems);
        sync_dissem_threadgroup_2<NVSHMEMI_THREADGROUP_THREAD>(start, stride, size, pSync,
                                                               sync_counter);
    }
}

template <typename TYPE, rdxn_ops_t OP>
__device__ static inline void gpu_rdxn_on_demand(nvshmem_team_t team, TYPE *dest,
                                                 const TYPE *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    volatile long *pSync = nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = nvshmemi_team_get_sync_counter(teami);

    gpu_rdxn_on_demand_2<TYPE, OP>(start, stride, size, dest, source, nelems, pWrk, pSync,
                                   sync_counter);
}

/* pWrk usage - (k - 1) * nreduce for step 1
              - k * step2_nphases * nreduce for receiving step 2 data
              - step2_nphases * nreduce for sending data of each phase */
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ static inline void gpu_rdxn_recexch_threadgroup(nvshmem_team_t team, TYPE *dst,
                                                           const TYPE *source, size_t nreduce) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    volatile long *pSync = (volatile long *)nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = (volatile long *)nvshmemi_team_get_sync_counter(teami);
    const int step1_sendto = teami->reduce_recexch.step1_sendto;
    const int step1_nrecvs = teami->reduce_recexch.step1_nrecvs;
    const int *step1_recvfrom = teami->reduce_recexch.step1_recvfrom;
    const int step2_nphases = teami->reduce_recexch.step2_nphases;
    int **step2_nbrs = teami->reduce_recexch.step2_nbrs;
    const int rank = nvshmemi_device_state_d.mype;
    const int k = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_recexch_kval;

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int in_step2 = (step1_sendto == -1); /* whether this rank participates in Step 2 */

    if (in_step2 == 1) {
        for (size_t i = myIdx; i < nreduce; i += groupSize) {
            dst[i] = source[i];
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    if (in_step2 == 0) {
        size_t offset = (step1_sendto - rank - 1) * nreduce;
        nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(pWrk + offset, source, nreduce, step1_sendto);
        if (!myIdx) {
            nvshmem_fence();
            nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                              step1_sendto);
        }
    } else if (step1_nrecvs != 0) {
        for (int i = 0; i < step1_nrecvs; i += 1) {
            nvshmem_long_wait_until((long *)pSync + step1_recvfrom[i], NVSHMEM_CMP_GE,
                                    sync_counter[0]);
            size_t offset = (rank - step1_recvfrom[i] - 1) * nreduce;
            gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dst, (pWrk + offset), dst, nreduce);
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    /* Step 2 */
    if (in_step2) {
        size_t send_offset = (k - 1) * nreduce + k * step2_nphases * nreduce;
        size_t recv_offset = (k - 1) * nreduce;
        for (int phase = 0; phase < step2_nphases; phase++) {
            int num_small = k - 1;
            for (int i = 0; i < k - 1; i++) {
                if (step2_nbrs[phase][i] > rank) {
                    num_small = i;
                    break;
                }
            }
            /* copy the data to end of pWrk that can be used as source for puts
                while we use dst for reduction */
            for (size_t i = myIdx; i < nreduce; i += groupSize) {
                pWrk[send_offset + phase * nreduce + i] = dst[i];
            }
            nvshmemi_threadgroup_sync<SCOPE>();
            for (int i = 0; i < k - 1; i++) {
                size_t offset = recv_offset + k * phase * nreduce + num_small * nreduce;
                nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(pWrk + offset,
                                                          pWrk + send_offset + phase * nreduce,
                                                          nreduce, step2_nbrs[phase][i]);
            }
            if (!myIdx) nvshmem_fence();
            nvshmemi_threadgroup_sync<SCOPE>();
            for (int i = myIdx; i < k - 1; i += groupSize) {
                nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                                  step2_nbrs[phase][i]);
            }

            for (int i = 0; i < k - 1; i += 1) {
                nvshmem_uint64_wait_until((uint64_t *)(pSync + step2_nbrs[phase][i]),
                                          NVSHMEM_CMP_GE, sync_counter[0]);
                int offset = recv_offset + k * phase * nreduce;
                if (step2_nbrs[phase][i] < rank)
                    offset += i * nreduce;
                else
                    offset += (i + 1) * nreduce;
                gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dst, (pWrk + offset), dst, nreduce);
            }
            /*nvshmem_quiet(); */ /*wait for my puts to complete */
        }
    }

    /* Step 3 */
    if (step1_nrecvs > 0) {
        for (int i = 0; i < step1_nrecvs; i++) {
            nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(dst, dst, nreduce, step1_recvfrom[i]);
        }
        if (!myIdx) nvshmem_fence();
        nvshmemi_threadgroup_sync<SCOPE>();
        for (int i = myIdx; i < step1_nrecvs; i += groupSize) {
            nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                              step1_recvfrom[i]);
        }
    } else if (step1_sendto != -1) {
        if (!myIdx)
            nvshmem_uint64_wait_until((uint64_t *)(pSync + step1_sendto), NVSHMEM_CMP_GE,
                                      sync_counter[0]);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    if (!myIdx) sync_counter[0] += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_head_check_op_threadgroup(T *dest, T *src, T *actual_src,
                                                            size_t nelems) {
    int i, k;
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    volatile uint32_t *header = NULL;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>;
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE tmp;
    uint32_t *tmp_ptr = (uint32_t *)&tmp;
    uint32_t *payload = NULL;
    for (i = myIdx; i < nelems; i += groupSize) {
        for (k = 0; k < subelems; k++) {
            payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k);
            header = (uint32_t *)payload + 1;
            while (1 != *header)
                ;
            *header = 0;
            *(tmp_ptr + k) = *payload;
        }
        dest[i] = perform_gpu_rdxn<T, OP>(actual_src[i], tmp);
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_head_checkall_op_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                               TYPE *src, TYPE *actual_src,
                                                               size_t nelems) {
    int i, j, k;
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    volatile uint32_t *header = NULL;
    TYPE tmp;
    uint32_t *tmp_ptr = (uint32_t *)&tmp;
    uint32_t *payload = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    for (j = (my_active_set_pe - 1); j >= 0; j--) {
        for (i = myIdx; i < nelems; i += groupSize) {
            for (k = 0; k < subelems; k++) {
                payload =
                    (uint32_t *)((uint64_t *)src + (i * subelems) + k + (nelems * subelems * j));
                header = (uint32_t *)payload + 1;
                while (1 != *header)
                    ;
                *header = 0;
                *(tmp_ptr + k) = *payload;
            }
            dest[i] = perform_gpu_rdxn<TYPE, OP>(src_ptr[i], tmp);
        }
        src_ptr = dest;
    }
    for (j = size - 1; j > my_active_set_pe; j--) {
        for (i = myIdx; i < nelems; i += groupSize) {
            for (k = 0; k < subelems; k++) {
                payload =
                    (uint32_t *)((uint64_t *)src + (i * subelems) + k + (nelems * subelems * j));
                header = (uint32_t *)payload + 1;
                while (1 != *header)
                    ;
                *header = 0;
                *(tmp_ptr + k) = *payload;
            }
            dest[i] = perform_gpu_rdxn<TYPE, OP>(src_ptr[i], tmp);
        }
        src_ptr = dest;
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup_p2p_get(nvshmem_team_t team, TYPE *x,
                                                                    TYPE *y, int next_rank, TYPE *z,
                                                                    size_t nelems) {
    int i;
    int group_nelems = ((nelems / groupSize) * groupSize);
    int excess = nelems - group_nelems;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    for (i = myIdx; i < group_nelems; i += groupSize) {
        nvshmemi_get_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
            (TYPE *)((TYPE *)pWrk + myIdx), (TYPE *)((TYPE *)y + i), 1, next_rank);
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], pWrk[myId]);
    }
    if (excess) {
        if (i < nelems) {
            nvshmemi_get_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
                (TYPE *)((TYPE *)pWrk + myIdx), (TYPE *)((TYPE *)y + i), 1, next_rank);
        }
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        if (i < nelems) {
            z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], pWrk[myIdx]);
        }
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup_p2p_put(TYPE *x, TYPE *y, int next_rank,
                                                                    TYPE *z, size_t offset,
                                                                    size_t nelems) {
    int i;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int group_nelems = ((nelems / groupSize) * groupSize);
    int excess = nelems - group_nelems;
    for (i = myIdx; i < group_nelems; i += groupSize) {
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
            (TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))),
            (const TYPE *)((TYPE *)y + i), 1, next_rank);
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
    }
    offset++;
    if (excess) {
        if (i < nelems) {
            nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
                (TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))),
                (const TYPE *)((TYPE *)y + i), 1, next_rank);
        }
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        if (i < nelems) {
            z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
        }
    }
    offset++;
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup_p2p_put_direct(TYPE *x, TYPE *y,
                                                                           int next_rank, TYPE *z,
                                                                           size_t offset,
                                                                           size_t nelems) {
    int i;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int group_nelems = ((nelems / groupSize) * groupSize);
    int excess = nelems - group_nelems;
    for (i = myIdx; i < group_nelems; i += groupSize) {
        *((TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize)))) =
            *((TYPE *)((TYPE *)y + i));
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
    }
    offset++;
    if (excess) {
        if (i < nelems) {
            *((TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize)))) =
                *((TYPE *)((TYPE *)y + i));
        }
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        if (i < nelems) {
            z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
        }
    }
    offset++;
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    base =
        (char *)((void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                               nvshmemi_device_state_d.mype));
    src_offset = ((char *)source - base);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    next_offset = src_offset;
    peer_base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank));
    peer_source = peer_base + next_offset;
    gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>((void *)source, peer_source, dest, nreduce);

    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        next_offset = src_offset;
        peer_base = (char *)((void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank));
        peer_source = peer_base + next_offset;
        gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dest, peer_source, dest, nreduce);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

// barrier limited
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_get_bar(nvshmem_team_t team, TYPE *dest,
                                                                    const TYPE *source,
                                                                    size_t nreduce) {
    int next_rank = -1;
    size_t src_offset;
    size_t next_offset;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    gpu_linear_reduce_threadgroup_p2p_get<TYPE, OP, SCOPE>(team, source, source, next_rank, dest,
                                                           nreduce);
    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        gpu_linear_reduce_threadgroup_p2p_get<TYPE, OP, SCOPE>(team, dest, source, next_rank, dest,
                                                               nreduce);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

// barrier limited
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline void nvshmemi_gpu_rdxn_threadgroup_put_bar(nvshmem_team_t team, TYPE *dest,
                                                         const TYPE *source, size_t nreduce) {
    int next_rank = -1;
    int counter = 0;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    gpu_linear_reduce_threadgroup_p2p_put<TYPE, OP, SCOPE>(source, source, next_rank, dest, counter,
                                                           nreduce);

    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        gpu_linear_reduce_threadgroup_p2p_put<TYPE, OP, SCOPE>(dest, source, next_rank, dest,
                                                               counter, nreduce);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_rdxn_segment_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                           const TYPE *source, size_t nelems) {
    int type_size = sizeof(TYPE);
    size_t msg_len = nelems * type_size;
    int next_rank = -1;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    TYPE *op1 = NULL, *op2 = NULL;
    int i;
    size_t j;
    volatile TYPE *tmp_operand;
    size_t remainder = 0;
    size_t rnds_floor = 0;
    size_t offset = 0;
    int pe_offset = 0;
    int pes_per_round = 0;
    int round = 0;
    size_t exchange_size = 0;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    size_t nvshm_gpu_rdxn_seg_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE;

    tmp_operand = (TYPE *)pWrk;
    nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>((TYPE *)dest, (const TYPE *)source, nelems,
                                              nvshmemi_device_state_d.mype);

    rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;
    remainder = msg_len % nvshm_gpu_rdxn_seg_size;

    for (j = 0; j < rnds_floor; j++) {
        exchange_size = nvshm_gpu_rdxn_seg_size;
        for (i = 1; i < size; i++) {
            next_rank = start + ((my_active_set_pe + i) % size) * stride;
            nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>((TYPE *)tmp_operand,
                                                      (const TYPE *)source + offset,
                                                      (exchange_size / sizeof(TYPE)), next_rank);
            nvshmemi_barrier_threadgroup<SCOPE>(team);
            op1 = (TYPE *)dest + offset;
            op2 = (TYPE *)tmp_operand;
            gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op1,
                                                           (exchange_size / sizeof(TYPE)));
            nvshmemi_sync_threadgroup<SCOPE>(team);
        }
        offset += (exchange_size / sizeof(TYPE));
    }
    if (remainder != 0) {
        exchange_size = remainder;
        pes_per_round = nvshm_gpu_rdxn_seg_size / remainder;
        pe_offset = 1;
        do {
            round = 0;
            for (i = pe_offset; ((round < pes_per_round) && (i < size)); i++) {
                next_rank = start + ((my_active_set_pe + i) % size) * stride;
                nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(
                    (TYPE *)((TYPE *)tmp_operand + (round * (exchange_size / sizeof(TYPE)))),
                    (TYPE *)source + offset, (exchange_size / sizeof(TYPE)), next_rank);
                round++;
                pe_offset++;
            }
            nvshmemi_barrier_threadgroup<SCOPE>(team);
            for (i = 0; i < round; i++) {
                op1 = (TYPE *)dest + offset;
                op2 = (TYPE *)((TYPE *)tmp_operand + (i * (exchange_size / sizeof(TYPE))));
                gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op1,
                                                               (exchange_size / sizeof(TYPE)));
            }
            nvshmemi_sync_threadgroup<SCOPE>(team);
        } while (pe_offset < size);
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_put_bar_direct(nvshmem_team_t team,
                                                                           TYPE *dest,
                                                                           const TYPE *source,
                                                                           size_t nreduce) {
    int next_rank = -1;
    size_t src_offset = -1;
    size_t next_offset = -1;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    int counter = 0;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    int i;

    base =
        (char *)((void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                               nvshmemi_device_state_d.mype));
    src_offset = ((char *)source - base);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    next_offset = src_offset;
    peer_base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank));
    peer_source = peer_base + next_offset;
    gpu_linear_reduce_threadgroup_p2p_put_direct<TYPE, OP, SCOPE>(source, peer_source, next_rank,
                                                                  dest, counter, nreduce);

    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        next_offset = src_offset;
        peer_base = (char *)((void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + next_rank));
        peer_source = peer_base + next_offset;
        gpu_linear_reduce_threadgroup_p2p_put_direct<TYPE, OP, SCOPE>(dest, peer_source, next_rank,
                                                                      dest, counter, nreduce);
    }
    nvshmemi_threadgroup_sync();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putring(nvshmem_team_t team, TYPE *dest,
                                                                    const TYPE *source,
                                                                    size_t nelems) {
    int next_rank = -1;
    int prev_rank = -1;
    int i;
    size_t j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint32_t tmp[2];
    uint64_t *tmp_rdxn = pWrk;
    uint64_t *tmp_rdxn_2 = &tmp_rdxn[1];
    uint32_t payld;
    volatile uint32_t *my_notif_ptr = NULL;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    tmp[1] = 1;
    next_rank =
        (nvshmemi_mype_d != (PE_end - stride)) ? (nvshmemi_device_state_d.mype + stride) : start;
    prev_rank =
        (nvshmemi_mype_d != start) ? (nvshmemi_device_state_d.mype - stride) : (PE_end - stride);
    my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx);

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        *tmp_rdxn = *((uint64_t *)tmp);
        nvshmemi_signal_for_barrier<long>((long *)(pWrk + j), *(long *)tmp_rdxn, next_rank);
    }
    gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    /* sync needed on volta (intermittent hangs seen otherwise) */
    nvshmem_threadgroup_sync<SCOPE>();

    for (i = 1; i < (size - 1); i++) {
        __threadfence_system();
        /* Don't want notification to overtake data transfer */
        *tmp_rdxn_2 = 1;
        nvshmemi_signal_for_barrier<long>((long *)(pWrk + (nelems * subelems) + myIdx),
                                          *(long *)tmp_rdxn_2, prev_rank);
        while (1 != *my_notif_ptr)
            ;
        *my_notif_ptr = 0;
        for (j = myIdx; j < nelems * subelems; j += groupSize) {
            payld = *((uint32_t *)dest + j);
            tmp[0] = payld;
            *tmp_rdxn = *((uint64_t *)tmp);
            nvshmemi_signal_for_barrier<long>((long *)(pWrk + j), *(long *)tmp_rdxn, next_rank);
        }
        gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putring_direct(nvshmem_team_t team,
                                                                           TYPE *dest,
                                                                           const TYPE *source,
                                                                           size_t nelems) {
    size_t offset;
    char *notif_pwrk_dest;
    char *round_pwrk_dest;
    int next_rank = -1;
    int prev_rank = -1;
    int i;
    size_t j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint32_t tmp[2];
    uint32_t payld;
    uint32_t *notif_ptr = NULL;
    volatile uint32_t *my_notif_ptr = NULL;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    tmp[1] = 1;
    offset = (char *)pWrk -
             (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                            nvshmemi_mype_d));
    my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx);
    next_rank = (nvshmemi_device_state_d.mype != (PE_end - stride))
                    ? (nvshmemi_device_state_d.mype + stride)
                    : start;
    round_pwrk_dest =
        (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                       next_rank)) +
        offset;
    prev_rank = (nvshmemi_device_state_d.mype != start) ? (nvshmemi_device_state_d.mype - stride)
                                                        : (PE_end - stride);
    notif_pwrk_dest =
        (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                       prev_rank)) +
        offset;
    notif_ptr =
        (uint32_t *)((uint32_t *)((uint64_t *)notif_pwrk_dest + (nelems * subelems)) + myIdx);

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);
    }
    gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    /* sync needed on volta (intermittent hangs seen otherwise) */
    nvshmemi_threadgroup_sync<SCOPE>();

    for (i = 1; i < (size - 1); i++) {
        __threadfence_system();
        /* Don't want notification to overtake data transfer */
        *notif_ptr = 1;
        while (1 != *my_notif_ptr)
            ;
        *my_notif_ptr = 0;
        for (j = myIdx; j < nelems * subelems; j += groupSize) {
            payld = *((uint32_t *)dest + j);
            tmp[0] = payld;
            *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);
        }
        gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putall(nvshmem_team_t team, TYPE *dest,
                                                                   const TYPE *source,
                                                                   size_t nelems) {
    int i;
    size_t j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint64_t *tmp_rdxn = pWrk;
    uint32_t tmp[2];
    uint32_t payld;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp[1] = 1;

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        *tmp_rdxn = *((uint64_t *)tmp);
        for (i = start; i < nvshmemi_device_state_d.mype; i += stride) {
            nvshmemi_signal_for_barrier<long>(
                (long *)(pWrk + j + (nelems * subelems * my_active_set_pe)), *(long *)tmp_rdxn, i);
        }
        for (i = nvshmemi_device_state_d.mype + stride; i < PE_end; i += stride) {
            nvshmemi_signal_for_barrier<long>(
                (long *)(pWrk + j + (nelems * subelems * my_active_set_pe)), *(long *)tmp_rdxn, i);
        }
    }
    gpu_head_checkall_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    __threadfence();
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putall_direct(nvshmem_team_t team,
                                                                          TYPE *dest, TYPE *source,
                                                                          size_t nelems) {
    size_t offset;
    char *round_pwrk_dest;
    int i, j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint32_t tmp[2];
    uint32_t payld;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp[1] = 1;
    offset = (char *)pWrk -
             (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                            nvshmemi_device_state_d.mype));

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        for (i = nvshmemi_device_state_d.mype + stride; i < PE_end; i += stride) {
            round_pwrk_dest =
                (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                               i)) +
                offset;
            *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =
                *((uint64_t *)tmp);
        }
        for (i = start; i < nvshmemi_device_state_d.mype; i += stride) {
            round_pwrk_dest =
                (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base +
                               i)) +
                offset;
            *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =
                *((uint64_t *)tmp);
        }
    }
    gpu_head_checkall_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    __threadfence();
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_hierarchical_fcollect_threadgroup(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
    nvshmemi_team_t *teami_node = nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX];
    nvshmemi_team_t *teami_same_mype_node =
        nvshmemi_device_state_d.team_pool[NVSHMEMX_TEAM_SAME_MYPE_NODE];
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    if (!myIdx) { /* Only one thread should increment rdxn_count */
        teami_node->rdxn_count++;
        teami_same_mype_node->rdxn_count++;
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami_node, REDUCE);
    if (teami_node->size >= 2)
        nvshmemi_fcollect_threadgroup<TYPE, SCOPE>(NVSHMEMX_TEAM_NODE, pWrk, source, nreduce);
    else
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nreduce * sizeof(TYPE));

    if (teami_node->size >= 2) {
        for (int j = myIdx; j < nreduce; j += groupSize) {
            gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                pWrk + j, pWrk + nreduce + j, dest + j, 1);
            for (int i = 2; i < teami_node->size; i++) {
                gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                    pWrk + i * nreduce + j, dest + j, dest + j, 1);
            }
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    if (teami_same_mype_node->size >= 2) {
        pWrk = (TYPE *)nvshmemi_team_get_psync(teami_same_mype_node, REDUCE);
        nvshmemi_fcollect_threadgroup<TYPE, SCOPE>(NVSHMEMX_TEAM_SAME_MYPE_NODE, pWrk, dest,
                                                   nreduce);
        for (int j = myIdx; j < nreduce; j += groupSize) {
            gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                (TYPE *)pWrk + j, (TYPE *)pWrk + nreduce + j, dest + j, 1);
            for (int i = 2; i < teami_same_mype_node->size; i++) {
                gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                    (TYPE *)pWrk + i * nreduce + j, dest + j, dest + j, 1);
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                            const TYPE *source, size_t nreduce) {
#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
    nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct<TYPE, OP, SCOPE>(team, dest, source,
                                                                        nreduce);
#else
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    size_t pwrk_req_sz_allgather = ((subelems * nreduce) * sizeof(uint64_t)) * size;
    size_t wrk_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);
    if (subelems && (pwrk_req_sz_allgather <= wrk_size)) {
        nvshmemi_gpu_rdxn_threadgroup_putall_direct<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct<TYPE, OP, SCOPE>(team, dest, source,
                                                                            nreduce);
    }
#endif
#else
#ifdef NVSHMEM_DISABLE_COLL_POLL
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int k = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_recexch_kval;
    if (start == 0 && stride == 1 && size == nvshmemi_device_state_d.npes && sizeof(TYPE) >= 4 &&
        nreduce % 2 == 0 &&
        NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * sizeof(long) >= size * nreduce * sizeof(TYPE) &&
        nvshmemi_device_state_d.fcollect_ll_threshold >= nreduce * sizeof(TYPE) &&
        SCOPE == NVSHMEMI_THREADGROUP_BLOCK)
        nvshmemi_gpu_rdxn_hierarchical_fcollect_threadgroup<TYPE, OP, SCOPE>(team, dest, source,
                                                                             nreduce);
    else if (start == 0 && stride == 1 && size == nvshmemi_device_state_d.npes &&
             NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE >=
                 ((k - 1) * nreduce + k * teami->reduce_recexch.step2_nphases * nreduce +
                  teami->reduce_recexch.step2_nphases * nreduce)) {
        gpu_rdxn_recexch_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        gpu_rdxn_segment_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    }
#else
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    size_t pwrk_req_sz_allgather = ((subelems * nreduce) * sizeof(uint64_t)) * size;
    int groupSize = nvshmemi_threadgroup_size<SCCOPE>();
    size_t pwrk_req_sz_ring =
        ((subelems * nreduce) * sizeof(uint64_t)) + (groupSize * sizeof(uint32_t));
    size_t wrk_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);
    if (subelems && pwrk_req_sz_allgather <= wrk_size) {
        nvshmemi_gpu_rdxn_threadgroup_putall<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else if (subelems && pwrk_req_sz_ring <= wrk_size) {
        nvshmemi_gpu_rdxn_threadgroup_putring<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        nvshmemi_gpu_rdxn_threadgroup_put_bar<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    }
#endif
#endif
}

/* This is the entry function for any rdxn collective op - host, on-stream, device
   There is only one exception - nvshmemi_reduce_kernel that is directly calling
   a specific reduction algorithm. That is a special need for team creation */
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ inline void nvshmemi_reduce_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                   const TYPE *source, size_t nreduce) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if (!myIdx) /* Only one thread should increment rdxn_count */
        nvshmemi_device_state_d.team_pool[team]->rdxn_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemi_gpu_rdxn_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
}

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                   \
        nvshmem##SC_PREFIX##_##TYPENAME##_##OP##_reduce##SC_SUFFIX(                                \
            nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {                 \
        nvshmemi_reduce_threadgroup<TYPE, RDXN_OPS_##OP, nvshmemi_threadgroup_##SC>(               \
            team, dest, source, nreduce);                                                          \
        return 0;                                                                                  \
    }

#define DEFN_NVSHMEM_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)                                  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                            \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, and)               \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                            \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, or)                \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                            \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, xor)               \
                                                                                                   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(                                           \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, max)               \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(                                           \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, min)               \
                                                                                                   \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, \
                                                     SC, SC_SUFFIX, SC_PREFIX, sum)                \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, \
                                                     SC, SC_SUFFIX, SC_PREFIX, prod)

DEFN_NVSHMEM_REDUCE_THREADGROUP(thread, , );
DEFN_NVSHMEM_REDUCE_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEM_REDUCE_THREADGROUP(block, _block, x);
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP
#undef DEFN_NVSHMEM_REDUCE_THREADGROUP

static __device__ inline void nvshmemi_double2_maxloc_reduce_alltoall_block(nvshmem_team_t team,
                                                                            double2 *dest,
                                                                            const double2 *source) {
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->rdxn_count += 1;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, REDUCE);

    nvshmemi_packLL<double2, SCOPE>((uint64_t *)(pWrk), source, 1, ll_flag);
    nvshmemi_threadgroup_sync<SCOPE>();
    const int my_pe = nvshmem_team_my_pe(team);
    const int n_pes = nvshmem_team_n_pes(team);
    for (int i = myIdx + 1; i < n_pes; i += groupSize) {
        int peer = (my_pe + i) % n_pes;
        size_t offset = 2 * sizeof(double2) * (my_pe + 2);
        nvshmemi_put_nbi_threadgroup<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
            (uint64_t *)(pWrk + offset), (uint64_t *)(pWrk), sizeof(double2) / sizeof(uint32_t),
            nvshmem_team_translate_pe(team, peer, NVSHMEM_TEAM_WORLD));
    }

    if (!myIdx) {
        dest[0] = source[0];
        for (int i = 1; i < n_pes; i++) {
            int peer = (my_pe + i) % n_pes;
            size_t offset = 2 * sizeof(double2) * (peer + 2);
            nvshmemi_recvLL<double2, NVSHMEMI_THREADGROUP_THREAD>(
                (double2 *)(pWrk + 2 * sizeof(double2)), (uint64_t *)(pWrk + offset), 1, ll_flag);
            dest[0] = perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(
                dest[0], *((double2 *)(pWrk + 2 * sizeof(double2))));
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
#undef SCOPE
}

static __device__ inline void nvshmemi_double2_maxloc_rooted_reduce_flat_block(
    nvshmem_team_t team, double2 *dest, const double2 *source) {
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->rdxn_count += 1;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, REDUCE);

    if (nvshmem_team_my_pe(team) != 0) {
        nvshmemi_packLL<double2, SCOPE>((uint64_t *)(pWrk), source, 1, ll_flag);
        size_t offset = 2 * sizeof(double2) * nvshmem_team_my_pe(team);
        nvshmemi_put_nbi_threadgroup<uint64_t, SCOPE>(
            (uint64_t *)(pWrk + offset), (uint64_t *)(pWrk), sizeof(double2) / sizeof(uint32_t),
            nvshmem_team_translate_pe(team, 0, NVSHMEM_TEAM_WORLD));
    } else {
        dest[0] = source[0];
        if (!myIdx) {
            for (int i = 1; i < teami->size; i += 1) {
                size_t offset = 2 * sizeof(double2) * i;
                nvshmemi_recvLL<double2, NVSHMEMI_THREADGROUP_THREAD>(
                    (double2 *)pWrk, (uint64_t *)(pWrk + offset), 1, ll_flag);
                dest[0] = perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(dest[0], *(double2 *)pWrk);
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
#undef SCOPE
}

/* double2 MAXLOC implementation */
static __device__ inline int nvshmemx_double2_maxloc_reduce_block(nvshmem_team_t team,
                                                                  double2 *dest,
                                                                  const double2 *source,
                                                                  size_t nreduce) {
#ifdef NVSHMEM_DEBUG
    assert(nreduce == 1);
#endif
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    switch (nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_algo) {
        case 1: /*  Alltoall algorithm */
            nvshmemi_double2_maxloc_reduce_alltoall_block(team, dest, source);
            break;
        case 2: /* Topo-unaware: Flat reduce + Flat bcast */
            nvshmemi_double2_maxloc_rooted_reduce_flat_block(team, dest, source);
            nvshmemi_bcast_internode_tree_threadgroup<double2, SCOPE>(team, dest, dest, 1, 0,
                                                                      nvshmem_team_n_pes(team) - 1);
            break;
        case 3: /* Topo aware two-level flat reduce + Topo aware two-level tree bcast */
            if (teami->is_team_node || teami->is_team_same_mype_node) {
                nvshmemi_double2_maxloc_rooted_reduce_flat_block(team, dest, source);
            } else {
                nvshmemi_double2_maxloc_rooted_reduce_flat_block(teami->team_node, dest, source);
                if (nvshmem_team_my_pe(teami->team_node) == 0) {
                    nvshmemi_double2_maxloc_rooted_reduce_flat_block(teami->team_same_mype_node,
                                                                     dest, dest);
                }
            }
            nvshmemi_bcast_hierarchical_threadgroup<double2, SCOPE>(team, dest, dest, 1, 0);
            break;
        case 4: /* Topo aware two-level flat reduce + Topo aware two-level tree bcast */
            if (teami->is_team_node || teami->is_team_same_mype_node) {
                nvshmemi_double2_maxloc_reduce_alltoall_block(team, dest, source);
            } else {
                nvshmemi_double2_maxloc_reduce_alltoall_block(teami->team_node, dest, source);
                nvshmemi_double2_maxloc_reduce_alltoall_block(teami->team_same_mype_node, dest,
                                                              dest);
            }
            break;
        default:
            assert(0);
            break;
    }
#undef SCOPE
    return 0;
}

#endif /* __CUDA_ARCH__ */

/* This is a special kernel that is launched only with
one thread and is used during team creation in nvshmemi_team_plit_strided fn */
template <typename TYPE, rdxn_ops_t OP>
__global__ void nvshmemi_reduce_kernel(int start, int stride, int size, TYPE *dst,
                                       const TYPE *source, size_t nreduce, TYPE *pWrk,
                                       volatile long *pSync, volatile long *sync_counter) {
#ifdef __CUDA_ARCH__
    gpu_rdxn_on_demand_2<TYPE, OP>(start, stride, size, dst, source, nreduce, pWrk, pSync,
                                   sync_counter);
#endif
}

#endif /* REDUCE_DEVICE_CUH */
