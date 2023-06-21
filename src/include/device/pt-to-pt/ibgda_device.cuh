/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_GIC_DEVICE_H_
#define _NVSHMEMI_GIC_DEVICE_H_

#include "infiniband/mlx5dv.h"

//#include "nvshmem_internal.h"
#include "nvshmemx_error.h"
#include "nvshmemi_util.h"
#include "utils_device.h"

#include <algorithm>

//#define NVSHMEM_IBGDA_DEBUG
//#define NVSHMEM_TIMEOUT_DEVICE_POLLING

#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE

#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
// These PTX optimizations are for GPU memory access only.
// Both data and NIC control objects must be in GPU memory.
#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
#endif

#define GIC_FULL_WARP 0xffffffffU
#define GIC_POLL_TIMEOUT 4000000000LLU

/* When we exceed a specific number of threads doing quiet
 * we end up with cache thrashing which causes a significant
 * perf hit. TODO: Tune this number for each supported arch.
 */
#define GIC_MAX_THREADS_PER_QUIET 32

// MLX5 accepts up to 2 GiB per command
#define GIC_MAX_TRANSFER_SIZE 2147483648LLU

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

/**
 * DO NOT use BSWAP(READ_ONCE(x)) as it could create a bug.
 * BSWAP is a pre-processor function. It will be unrolled to many READ_ONCE.
 */
#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

#ifdef NVSHMEM_IBGDA_DEBUG
struct mlx5_err_cqe_ex {
    uint8_t rsvd0[32];
    __be32 srqn;
    uint8_t rsvd1[16];
    uint8_t hw_err_synd;
    uint8_t hw_synd_type;
    uint8_t vendor_err_synd;
    uint8_t syndrome;
    __be32 s_wqe_opcode_qpn;
    __be16 wqe_counter;
    uint8_t signature;
    uint8_t op_own;
};
typedef struct mlx5_err_cqe_ex gic_mlx5_err_cqe_t;
#else
typedef struct mlx5_err_cqe gic_mlx5_err_cqe_t;
#endif

#define GIC_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define GIC_8_BYTE_EXT_AMO_OPMOD 0x09000000

typedef enum gic_mlx5_fm {
    GIC_MLX5_FM_NO_FENCE = 0,
    GIC_MLX5_FM_INITIATOR_SMALL_FENCE = 1 << 5,
    GIC_MLX5_FM_FENCE = 2 << 5,
    GIC_MLX5_FM_STRONG_ORDERING = 3 << 5,
    GIC_MLX5_FM_FENCE_AND_INITIATOR_SMALL_FENCE = 4 << 5
} gic_mlx5_fm_t;

enum {
    GIC_MLX5_OPCODE_DUMP = 0x23,
};

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) gic_ctrl_seg_t;

// The ext flag (in dqp_dct) must be set to disable.
typedef struct {
    __be64 dc_key;
    __be32 dqp_dct;
    uint8_t stat_rate_sl;
    uint8_t fl_mlid;
    __be16 rlid;
} __attribute__((__packed__)) __attribute__((__aligned__(4))) gic_half_av_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_half_av_seg_t) == 16, "sizeof(gic_half_av_seg_t) == 16 failed.");
#endif

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) gic_atomic_32_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_32_masked_fa_seg_t) == 16,
              "sizeof(gic_atomic_32_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t add_data;
    uint64_t field_boundary;
} __attribute__((__packed__)) gic_atomic_64_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_64_masked_fa_seg_t) == 16,
              "sizeof(gic_atomic_64_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint32_t swap_data;
    uint32_t compare_data;
    uint32_t swap_mask;
    uint32_t compare_mask;
} __attribute__((__packed__)) gic_atomic_32_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_32_masked_cs_seg_t) == 16,
              "sizeof(gic_atomic_32_masked_cs_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t swap;
    uint64_t compare;
} __attribute__((__packed__)) gic_atomic_64_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_64_masked_cs_seg_t) == 16,
              "sizeof(gic_atomic_64_masked_cs_seg_t) == 16 failed.");
#endif

#ifdef __CUDA_ARCH__

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
__device__ static inline uint64_t gic_query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
    return ret;
}
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */

__device__ static inline nvshmemi_gic_device_state_t *gic_get_state() {
    return &nvshmemi_gic_device_state_d;
}

__device__ static inline bool gic_is_rc_enabled() { return gic_get_state()->num_rc_per_pe > 0; }

// Prevent code reordering from both compiler and GPU
__device__ static inline void GIC_MFENCE() {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#else
    __threadfence_block();
#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE */
}

__device__ static inline void GIC_MEMBAR_NO_OPTIMIZATION() {
#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    __threadfence();
#else
    if (likely(gic_get_state()->nic_buf_on_gpumem))
        __threadfence();
    else
        __threadfence_system();
#endif /* NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */
}

__device__ static inline void GIC_MEMBAR() {
// st.release automatically adds membar in SASS.
#ifndef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE

#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    __threadfence();
#else
    if (likely(gic_get_state()->nic_buf_on_gpumem))
        __threadfence();
    else
        __threadfence_system();
#endif /* NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */

#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE */
}

__device__ static inline uint32_t gic_get_smid() {
    uint32_t smid;
    asm("mov.u32  %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ static inline uint32_t gic_get_ctaid() {
    return (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
}

template <typename T>
__device__ static inline void gic_store_relaxed(T *ptr, T val) {
    WRITE_ONCE(*ptr, val);
}

template <>
__device__ inline void gic_store_relaxed(uint8_t *ptr, uint8_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    uint16_t _val = val;
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(_val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void gic_store_relaxed(uint16_t *ptr, uint16_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void gic_store_relaxed(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void gic_store_relaxed(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline void gic_store_release(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline void gic_store_release(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

/**
 * DO NOT use BSWAP(gic_atomic_read(x)) as it could create a bug.
 * See the comment near READ_ONCE.
 */
__device__ static inline uint8_t gic_atomic_read(uint8_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return (uint8_t)ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline uint16_t gic_atomic_read(uint16_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline uint32_t gic_atomic_read(uint32_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline uint64_t gic_atomic_read(uint64_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline void gic_atomic_set(int *ptr, int val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline size_t gic_cal_transfer_size(size_t req_size, size_t lchunk_size,
                                                      size_t rchunk_size) {
    return NVSHMEMI_MIN(GIC_MAX_TRANSFER_SIZE,
                        NVSHMEMI_MIN(req_size, NVSHMEMI_MIN(rchunk_size, lchunk_size)));
}

template <threadgroup_t SCOPE>
__device__ static inline void gic_lock_acquire(int *lock) {
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0)
        while (atomicCAS(lock, 0, 1) == 1)
            ;  // Wait until we get the lock.

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        GIC_MFENCE();  // Prevent reordering before lock is acquired.

    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ static inline void gic_lock_release(int *lock) {
    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        GIC_MFENCE();  // Prevent reordering before lock is released.

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) gic_atomic_set(lock, 0);
}

// Multiple threads may update get_head concurrently.
// Only the latest one w.r.t. wqe_idx is important.
__device__ static inline void gic_update_get_head(nvshmemi_gic_device_qp_t *qp,
                                                  uint64_t new_get_head) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_head,
              (unsigned long long int)new_get_head);
}

__device__ static inline void gic_update_get_tail(nvshmemi_gic_device_qp_t *qp,
                                                  uint64_t new_get_tail) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_tail,
              (unsigned long long int)new_get_tail);
}

__device__ static inline void *gic_get_wqe_ptr(nvshmemi_gic_device_qp_t *qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1);
    return (void *)((uintptr_t)qp->tx_wq.wqe + (idx << MLX5_SEND_WQE_SHIFT));
}

template <bool support_half_av_seg, nvshmemi_gic_device_qp_type_t qp_type>
__device__ static inline uint16_t gic_sop_to_nwqes(uint8_t sop) {
    if (qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_RC) {
        switch (sop) {
            case MLX5_OPCODE_NOP:
            case GIC_MLX5_OPCODE_DUMP:
            case MLX5_OPCODE_RDMA_WRITE:
            case MLX5_OPCODE_RDMA_READ:
                return 1;
            case MLX5_OPCODE_ATOMIC_MASKED_CS:
            case MLX5_OPCODE_ATOMIC_MASKED_FA:
                // We patch with NOP when they consume one wqebb.
                // Only NOP wqe will emit CQ update.
                return 2;
#ifndef NVSHMEM_IBGDA_DEBUG
            // The CQ buffer is still invalid and we skip checking opcode == MLX5_CQE_INVALID
            case 0xff:
                return 0;
#endif
            case MLX5_OPCODE_ATOMIC_CS:
            case MLX5_OPCODE_ATOMIC_FA:
                // We patch the last wqe with NOP.
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unknown sop.\n");
#endif
                assert(0);
                return 0;
        }
    } else if (support_half_av_seg) {
        switch (sop) {
            case MLX5_OPCODE_NOP:
            case GIC_MLX5_OPCODE_DUMP:
            case MLX5_OPCODE_RDMA_WRITE:
            case MLX5_OPCODE_RDMA_READ:
                return 1;
            case MLX5_OPCODE_ATOMIC_CS:
            case MLX5_OPCODE_ATOMIC_FA:
            case MLX5_OPCODE_ATOMIC_MASKED_CS:
            case MLX5_OPCODE_ATOMIC_MASKED_FA:
                return 2;
#ifndef NVSHMEM_IBGDA_DEBUG
            // The CQ buffer is still invalid and we skip checking opcode == MLX5_CQE_INVALID
            case 0xff:
                return 0;
#endif
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unknown sop.\n");
#endif
                assert(0);
                return 0;
        }
    } else {
        switch (sop) {
            case MLX5_OPCODE_NOP:
            case GIC_MLX5_OPCODE_DUMP:
                return 1;
            case MLX5_OPCODE_RDMA_WRITE:
            case MLX5_OPCODE_RDMA_READ:
            case MLX5_OPCODE_ATOMIC_CS:
            case MLX5_OPCODE_ATOMIC_FA:
            case MLX5_OPCODE_ATOMIC_MASKED_CS:
            case MLX5_OPCODE_ATOMIC_MASKED_FA:
                return 2;
#ifndef NVSHMEM_IBGDA_DEBUG
            // The CQ buffer is still invalid and we skip checking opcode == MLX5_CQE_INVALID
            case 0xff:
                return 0;
#endif
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unknown sop.\n");
#endif
                assert(0);
                return 0;
        }
    }
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MAX_QP_DEPTH <= 32768,
              "static_assert(NVSHMEMI_GIC_MAX_QP_DEPTH <= 32768) failed");
#endif
template <bool support_half_av_seg>
__device__ static inline int gic_poll_cq(nvshmemi_gic_device_cq_t *cq, uint64_t idx, int *error) {
    int status = 0;
    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)cq->cqe;

    uint8_t opown;
    uint8_t opcode;
    uint16_t wqe_counter;
    uint16_t polled_cons_tail;
    uint8_t sop;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    uint64_t start = gic_query_globaltimer();
    uint64_t now;
#endif

    uint64_t cons_head;
    uint64_t cons_tail = gic_atomic_read(cq->cons_tail);
    uint16_t cons_tail_lo;
    uint64_t cons_tail_hi;

    assert(likely(cq->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI ||
                  cq->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_RC));

    if (unlikely(cons_tail >= idx)) return 0;

#ifdef NVSHMEM_IBGDA_DEBUG
    // We can skip opcode == MLX5_CQE_INVALID check because we have already
    // initialized the CQ buffer to 0xff. With the QP depth range we enforce,
    // cons_tail cannot progress unless wqe_counter read from the CQ buffer is
    // a valid value.
    do {
        opown = gic_atomic_read(&cqe64->op_own);
        opcode = opown >> 4;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        // TODO: Integrate timeout handler with the core NVSHMEM
        now = gic_query_globaltimer();
        if (unlikely(now - start > GIC_POLL_TIMEOUT)) {
            *error = -ETIME;
            printf(
                "[%d] gic_poll_cq timeout because of MLX5_CQE_INVALID:\n"
                "    cons_tail=%#lx, cons_head=%#lx, cqn=%#x\n"
                "    qpn=%#x\n"
                "    wqe_head=%#lx, wqe_tail=%#lx\n"
                "    while waiting for idx=%#lx.\n",
                nvshmemi_device_state_d.mype, cons_tail, gic_atomic_read(cq->cons_head), cq->cqn,
                cq->qpn, gic_atomic_read(cq->wqe_head), gic_atomic_read(cq->wqe_tail), idx);
            status = -1;
            goto out;
        }
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */
    } while (unlikely(opcode == MLX5_CQE_INVALID));

    // Prevent reordering of the opcode wait above
    GIC_MFENCE();
#endif /* NVSHMEM_IBGDA_DEBUG */

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    now = gic_query_globaltimer();
#endif

    do {
        cons_tail_lo = cons_tail & 0xffff;
        cons_tail_hi = cons_tail & 0xffffffffffff0000ULL;

        // In collapsed + overrun ignored CQ, the data in the buffer can be updated anytime by NIC.
        // So, we need to read wqe_counter and sop in an atomic fasion.
        uint64_t chunk_data = gic_atomic_read((uint64_t *)&cqe64->sop_drop_qpn);
        chunk_data = BSWAP64(chunk_data);

        wqe_counter = (chunk_data >> 16) & 0xffffULL;
        sop = chunk_data >> 56;
        uint16_t nwqes;
        if (cq->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI)
            nwqes = gic_sop_to_nwqes<support_half_av_seg, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>(sop);
        else
            nwqes = gic_sop_to_nwqes<support_half_av_seg, NVSHMEMI_GIC_DEVICE_QP_TYPE_RC>(sop);
        polled_cons_tail = wqe_counter + nwqes;

        cons_head = gic_atomic_read(cq->cons_head);

        if (cons_tail_lo != polled_cons_tail) {
            if (unlikely(cons_tail_lo > polled_cons_tail)) {
                // Handle potential wrap around
                uint64_t new_cons_tail_hi = cons_tail_hi + 0x10000ULL;
                if (likely(cons_head >= new_cons_tail_hi + polled_cons_tail))
                    cons_tail_hi = new_cons_tail_hi;
            }

            cons_tail_lo = polled_cons_tail;
            cons_tail = cons_tail_hi + cons_tail_lo;

            if (cons_tail <= cons_head) {
                // Other threads might update cons_tail concurrently.
                atomicMax((unsigned long long int *)cq->cons_tail,
                          (unsigned long long int)cons_tail);
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
                // Observed index change; so reset the clock.
                start = now;
#endif
            }
        }

        cons_tail = gic_atomic_read(cq->cons_tail);

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        // TODO: Integrate timeout handler with the core NVSHMEM
        now = gic_query_globaltimer();
        if (unlikely(now - start > GIC_POLL_TIMEOUT)) {
            *error = -ETIME;
#ifdef NVSHMEM_IBGDA_DEBUG
            printf(
                "[%d] gic_poll_cq timeout:\n"
                "    cons_tail=%#lx, cons_head=%#lx, chunk_data=%#lx, cqn=%#x\n"
                "    polled_cons_tail=%#x, qpn=%#x\n"
                "    wqe_head=%#lx, wqe_tail=%#lx\n"
                "    while waiting for idx=%#lx.\n",
                nvshmemi_device_state_d.mype, cons_tail, cons_head, chunk_data, cq->cqn,
                polled_cons_tail, cq->qpn, gic_atomic_read(cq->wqe_head),
                gic_atomic_read(cq->wqe_tail), idx);
#endif /* NVSHMEM_IBGDA_DEBUG */
            status = -1;
            goto out;
        }
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */
    } while (unlikely(cons_tail < idx));

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
out:
#endif
    // Prevent reordering of the idx wait above
    GIC_MFENCE();

    // NVSHMEM always treats CQE errors as fatal.
    // Even if this error doesn't belong to the CQE in cons_tail,
    // we will just report and terminate the process.
    opown = gic_atomic_read(&cqe64->op_own);
    opcode = opown >> 4;

    if (unlikely(opcode == MLX5_CQE_REQ_ERR)) {
        gic_mlx5_err_cqe_t *cqe_err = (gic_mlx5_err_cqe_t *)cqe64;
        *error = cqe_err->syndrome;
#ifdef NVSHMEM_IBGDA_DEBUG
        __be16 wqe_counter = gic_atomic_read(&cqe_err->wqe_counter);
        __be32 s_wqe_opcode_qpn = gic_atomic_read(&cqe_err->s_wqe_opcode_qpn);
        printf(
            "[%d] got completion with err:\n"
            "   syndrome=%#x, vendor_err_synd=%#x, hw_err_synd=%#x, hw_synd_type=%#x,\n"
            "   wqe_counter=%#x, s_wqe_opcode_qpn=%#x,\n"
            "   cqn=%#x, cons_tail=%#lx, cons_head=%#lx, idx=%#lx\n",
            nvshmemi_device_state_d.mype, cqe_err->syndrome, cqe_err->vendor_err_synd,
            cqe_err->hw_err_synd, cqe_err->hw_synd_type, BSWAP16(wqe_counter),
            BSWAP32(s_wqe_opcode_qpn), cq->cqn, cons_tail, cons_head, idx);
#endif /* NVSHMEM_IBGDA_DEBUG */
        status = -1;
    }

    // Prevent reordering of this function and subsequent instructions
    GIC_MFENCE();

    return status;
}

__device__ static inline void gic_write_dump_wqe(nvshmemi_gic_device_qp_t *qp, uint64_t laddr,
                                                 __be32 lkey, uint32_t bytes, uint16_t wqe_idx,
                                                 gic_mlx5_fm_t fm, void **out_wqes,
                                                 gic_ctrl_seg_t *out_ctrl_seg) {
    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_data_seg data_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[0];
    struct mlx5_wqe_data_seg *data_seg_ptr =
        (struct mlx5_wqe_data_seg *)((uintptr_t)out_wqes[0] + sizeof(*ctrl_seg_ptr));

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | 2);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE | fm;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | GIC_MLX5_OPCODE_DUMP);

    // out_ctrl_seg is in register and will eventually consumed by gic_ring_db.
    // WRITE_ONCE is not necessary here.
    *out_ctrl_seg = ctrl_seg;

    // wqe_ptr will not be consumed by GPU.
    // WRITE_ONCE ensures that compiler will not removed this code.
    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);
}

template <bool support_half_av_seg>
__device__ static inline void gic_write_rdma_write_wqe(nvshmemi_gic_device_qp_t *qp,
                                                       nvshmemi_gic_device_dct_t *dct,
                                                       uint64_t laddr, __be32 lkey, uint64_t raddr,
                                                       __be32 rkey, uint32_t bytes,
                                                       uint16_t wqe_idx, void **out_wqes,
                                                       gic_ctrl_seg_t *out_ctrl_seg) {
    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    size_t av_seg_size;
    int ds;

    if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 4;
            av_seg_size = sizeof(gic_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        } else {
            ds = 6;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
        }
    } else {
        ds = 3;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    }
    data_seg_ptr = (struct mlx5_wqe_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    *out_ctrl_seg = ctrl_seg;

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) gic_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);
}

template <bool support_half_av_seg>
__device__ static inline void gic_write_rdma_write_inl_wqe(
    nvshmemi_gic_device_qp_t *qp, nvshmemi_gic_device_dct_t *dct, const void *val, uint64_t raddr,
    __be32 rkey, uint32_t bytes, uint16_t wqe_idx, void **out_wqes, gic_ctrl_seg_t *out_ctrl_seg) {
    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_inl_data_seg *inl_seg_ptr;
    void *wqe_data_ptr;

    size_t av_seg_size;
    int ds;

    // Allow up to 12 bytes
    assert(likely(bytes <= 12));

    if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 4;
            av_seg_size = sizeof(gic_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        } else {
            ds = 6;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
        }
    } else {
        ds = 3;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)av_seg_ptr;
    }
    inl_seg_ptr =
        (struct mlx5_wqe_inl_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
    wqe_data_ptr = (void *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HTOBE32(bytes | MLX5_INLINE_SEG);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    *out_ctrl_seg = ctrl_seg;

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) gic_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)inl_seg_ptr;
    src = (uint32_t *)&inl_seg;
    for (int i = 0; i < sizeof(*inl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    memcpy(wqe_data_ptr, val, bytes);
}

/**
 * For DC, support only half av seg.
 * The header already consumes 1 wqebb and leaves 12 bytes for inline data.
 * The last wqebb is no-op.
 * One wqebb is 64 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - 12) / 64) + 2
 *
 * For RC
 * The header already consumes 1 wqebb and leaves 12 + 16 bytes for inline data.
 * The last wqebb is no-op.
 * One wqebb is 64 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - (12 + 16)) / 64) + 2
 */
template <typename T, nvshmemi_gic_device_qp_type_t qp_type>
__device__ static inline uint32_t gic_get_num_wqes_in_inl_combine_warp() {
    if (qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        // DC supports up to 16 DS WQE
        switch (sizeof(T)) {
            case 1:
            case 2:
                return 3;
            case 4:
                return 4;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    } else {
        // RC supports up to 64 DS WQE
        switch (sizeof(T)) {
            case 1:
            case 2:
                return 3;
            case 4:
                return 4;
            case 8:
                return 6;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    }
}

/**
 * For DC, support only half av seg.
 * The header already consumes 4 ds and leaves 12 bytes for inline data.
 * One ds is 16 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - 12) / 16) + 4
 *
 * For RC
 * The header already consumes 3 ds and leaves 12 bytes for inline data.
 * One ds is 16 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - 12) / 16) + 3
 */
template <typename T, nvshmemi_gic_device_qp_type_t qp_type>
__device__ static inline uint32_t gic_get_ds_in_inl_combine_warp() {
    if (qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        // DC supports up to 16 DS WQE
        switch (sizeof(T)) {
            case 1:
                return 6;
            case 2:
                return 8;
            case 4:
                return 12;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    } else {
        // DC supports up to 16 DS WQE
        switch (sizeof(T)) {
            case 1:
                return 5;
            case 2:
                return 7;
            case 4:
                return 11;
            case 8:
                return 19;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    }
}

template <typename T>
__device__ static inline void gic_write_rdma_write_inl_wqe_combine_warp(
    nvshmemi_gic_device_qp_t *qp, nvshmemi_gic_device_dct_t *dct, const T val, uint64_t _raddr,
    __be32 rkey, uint16_t wqe_idx, int my_tid, void **out_wqes, gic_ctrl_seg_t *out_ctrl_seg) {
    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_inl_data_seg *inl_seg_ptr;

    size_t av_seg_size;
    int ds;

    uint32_t bytes = sizeof(T);
    uint64_t raddr = _raddr - (my_tid * bytes);

    int remaining_size_for_data_in_first_wqebb;
    uint32_t nop_relative_wqe_idx;

    if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        ds = gic_get_ds_in_inl_combine_warp<T, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>();
        av_seg_size = sizeof(gic_half_av_seg_t);
        remaining_size_for_data_in_first_wqebb = 12;
        nop_relative_wqe_idx =
            gic_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>() - 1;
    } else {
        ds = gic_get_ds_in_inl_combine_warp<T, NVSHMEMI_GIC_DEVICE_QP_TYPE_RC>();
        av_seg_size = 0;
        remaining_size_for_data_in_first_wqebb = 28;
        nop_relative_wqe_idx =
            gic_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_GIC_DEVICE_QP_TYPE_RC>() - 1;
    }

    raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    inl_seg_ptr =
        (struct mlx5_wqe_inl_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HTOBE32((bytes * warpSize) | MLX5_INLINE_SEG);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    // ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    // This RDMA WRITE wqe will not get CQ update to avoid dynamic size calculation in poll_cq.
    // Instead, the NO-OP wqe (last one) will get CQ update because it is always 1 WQEBB.
    ctrl_seg.fm_ce_se = 0;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) gic_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)inl_seg_ptr;
    src = (uint32_t *)&inl_seg;
    for (int i = 0; i < sizeof(*inl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    uint32_t my_base_data_idx = my_tid * bytes;
    if (bytes <= 4) {
        T *wqe_data_ptr;
        if (my_base_data_idx < remaining_size_for_data_in_first_wqebb)
            wqe_data_ptr = (T *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr) + my_base_data_idx);
        else {
            uint32_t my_data_idx = my_base_data_idx - remaining_size_for_data_in_first_wqebb;
            int my_data_in_wqe_idx = my_data_idx / 64 + 1;
            my_data_idx &= (64 - 1);  // my_data_idx % 64
            wqe_data_ptr = (T *)((uintptr_t)out_wqes[my_data_in_wqe_idx] + my_data_idx);
        }
        gic_store_relaxed(wqe_data_ptr, val);
    } else {
        // wqe_data_ptr is 4-byte aligned but not 8-byte aligned.
        assert(likely(bytes == 8 && qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_RC));
        uint32_t *wqe_data_ptr;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t my_data_idx = my_base_data_idx + (i * 4);
            if (my_data_idx < remaining_size_for_data_in_first_wqebb)
                wqe_data_ptr =
                    (uint32_t *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr) + my_data_idx);
            else {
                uint32_t my_idx = my_data_idx - remaining_size_for_data_in_first_wqebb;
                int my_data_in_wqe_idx = my_idx / 64 + 1;
                my_idx &= (64 - 1);  // my_idx % 64
                wqe_data_ptr = (uint32_t *)((uintptr_t)out_wqes[my_data_in_wqe_idx] + my_idx);
            }
            gic_store_relaxed(wqe_data_ptr, *((uint32_t *)&val + i));
        }
    }

    wqe_idx += nop_relative_wqe_idx;
    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | 1);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_NOP);

    ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[nop_relative_wqe_idx];

    dst = (uint32_t *)ctrl_seg_ptr;
    src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    *out_ctrl_seg = ctrl_seg;
}

/**
 * For DCI with sizeof(T) == 8 only.
 * DC supports up to 16 DS WQE.
 * For sizeof(T) == 8, we split to two WQEs of inline size 8 * 16
 */
template <typename T>
__device__ static inline void gic_write_rdma_write_inl_wqe_combine_warp_for_dci_8B(
    nvshmemi_gic_device_qp_t *dci, nvshmemi_gic_device_dct_t *dct, const T val, uint64_t _raddr,
    __be32 rkey, uint16_t _wqe_idx, int my_tid, void **out_wqes, gic_ctrl_seg_t *out_ctrl_seg) {
    assert(likely(sizeof(T) == 8 && dci->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI));

    // base_tid = my_tid >= 16 ? 16 : 0;
    int base_tid = my_tid & (~0xF);

    // base_wqe_idx = base_tid / 4;
    int base_out_wqe_idx = base_tid >> 2;

    uint16_t wqe_idx = _wqe_idx + base_out_wqe_idx;

    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[base_out_wqe_idx];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_inl_data_seg *inl_seg_ptr;
    uint32_t *wqe_data_ptr;

    size_t av_seg_size;
    int ds = gic_get_ds_in_inl_combine_warp<uint32_t, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>();

    uint64_t raddr = _raddr - ((my_tid - base_tid) * 8);

    av_seg_size = sizeof(gic_half_av_seg_t);
    raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    inl_seg_ptr =
        (struct mlx5_wqe_inl_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HTOBE32((8 * warpSize / 2) | MLX5_INLINE_SEG);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((dci->qpn << 8) | ds);
    // ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    // This RDMA WRITE wqe will not get CQ update to avoid dynamic size calculation in poll_cq.
    // Instead, the NO-OP wqe (last one) will get CQ update because it is always 1 WQEBB.
    ctrl_seg.fm_ce_se = 0;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)av_seg_ptr;
    src = (uint32_t *)dct;
    for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)inl_seg_ptr;
    src = (uint32_t *)&inl_seg;
    for (int i = 0; i < sizeof(*inl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    for (int i = 0; i < 2; ++i) {
        uint32_t my_data_idx = ((my_tid - base_tid) * 2 + i) * 4;
        if (my_data_idx < 12)
            wqe_data_ptr =
                (uint32_t *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr) + my_data_idx);
        else {
            my_data_idx -= 12;
            int my_data_in_wqe_idx = my_data_idx / 64 + 1;
            my_data_idx &= (64 - 1);  // my_data_idx % 64
            wqe_data_ptr = (uint32_t *)((uintptr_t)out_wqes[my_data_in_wqe_idx + base_out_wqe_idx] +
                                        my_data_idx);
        }

        gic_store_relaxed(wqe_data_ptr, ((uint32_t *)&val)[i]);
    }

    uint32_t nop_relative_wqe_idx =
        gic_get_num_wqes_in_inl_combine_warp<uint32_t, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>() - 1;

    wqe_idx += nop_relative_wqe_idx;
    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((dci->qpn << 8) | 1);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_NOP);

    ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[nop_relative_wqe_idx + base_out_wqe_idx];

    dst = (uint32_t *)ctrl_seg_ptr;
    src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    *out_ctrl_seg = ctrl_seg;
}

template <bool support_half_av_seg>
__device__ static inline void gic_write_rdma_read_wqe(nvshmemi_gic_device_qp_t *qp,
                                                      nvshmemi_gic_device_dct_t *dct,
                                                      uint64_t laddr, __be32 lkey, uint64_t raddr,
                                                      __be32 rkey, uint32_t bytes, uint16_t wqe_idx,
                                                      void **out_wqes,
                                                      gic_ctrl_seg_t *out_ctrl_seg) {
    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    size_t av_seg_size;
    int ds;

    if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 4;
            av_seg_size = sizeof(gic_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        } else {
            ds = 6;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
        }
    } else {
        ds = 3;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    }
    data_seg_ptr = (struct mlx5_wqe_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_READ);

    *out_ctrl_seg = ctrl_seg;

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) gic_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);
}

template <bool support_half_av_seg>
__device__ static inline void gic_write_atomic_wqe(
    nvshmemi_gic_device_qp_t *qp, nvshmemi_gic_device_dct_t *dct, const void *val_1,
    const void *val_2, uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, uint32_t bytes,
    uint16_t wqe_idx, nvshmemi_amo_t amo_op, void **out_wqes, gic_ctrl_seg_t *out_ctrl_seg) {
    gic_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_atomic_seg atomic_seg_1;
    struct mlx5_wqe_atomic_seg atomic_seg_2;
    struct mlx5_wqe_data_seg data_seg;

    gic_ctrl_seg_t *ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_atomic_seg *atomic_seg_1_ptr;
    struct mlx5_wqe_atomic_seg *atomic_seg_2_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    size_t av_seg_size;
    int ds;

    if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 5;
            av_seg_size = sizeof(gic_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
            atomic_seg_1_ptr =
                (struct mlx5_wqe_atomic_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
            atomic_seg_2_ptr = (struct mlx5_wqe_atomic_seg *)out_wqes[1];
        } else {
            ds = 7;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
            atomic_seg_1_ptr =
                (struct mlx5_wqe_atomic_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
            atomic_seg_2_ptr = (struct mlx5_wqe_atomic_seg *)((uintptr_t)atomic_seg_1_ptr +
                                                              sizeof(*atomic_seg_1_ptr));
        }
    } else {
        ds = 4;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        atomic_seg_1_ptr =
            (struct mlx5_wqe_atomic_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
        atomic_seg_2_ptr =
            (struct mlx5_wqe_atomic_seg *)((uintptr_t)atomic_seg_1_ptr + sizeof(*atomic_seg_1_ptr));
    }
    data_seg_ptr = (struct mlx5_wqe_data_seg *)atomic_seg_2_ptr;

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    ctrl_seg = {
        0,
    };

    assert(likely(bytes == 4 || bytes == 8));
    switch (amo_op) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (gic_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32((uint32_t)1);
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (gic_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = HTOBE64((uint64_t)1);
                atomic_64_masked_fa_seg->field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (gic_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = 0;
                atomic_32_masked_cs_seg->compare_mask = 0;
                atomic_32_masked_cs_seg->swap_mask = UINT32_MAX;
            } else {
                ++ds;
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                    (gic_atomic_64_masked_cs_seg_t *)&atomic_seg_1;
                atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_data_seg->compare = 0;

                gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                    (gic_atomic_64_masked_cs_seg_t *)&atomic_seg_2;
                atomic_64_masked_cs_mask_seg->swap = UINT64_MAX;
                atomic_64_masked_cs_mask_seg->compare = 0;

                if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI)
                    data_seg_ptr =
                        (struct mlx5_wqe_data_seg *)((uintptr_t)atomic_seg_2_ptr +
                                                     sizeof(*atomic_64_masked_cs_mask_seg));
                else
                    data_seg_ptr = (struct mlx5_wqe_data_seg *)out_wqes[1];
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (gic_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (gic_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_fa_seg->field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (gic_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = 0;
                atomic_32_masked_cs_seg->compare_mask = 0;
                atomic_32_masked_cs_seg->swap_mask = HTOBE32(~(*(uint32_t *)val_1));
            } else {
                ++ds;
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                    (gic_atomic_64_masked_cs_seg_t *)&atomic_seg_1;
                atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_data_seg->compare = 0;

                gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                    (gic_atomic_64_masked_cs_seg_t *)&atomic_seg_2;
                atomic_64_masked_cs_mask_seg->swap = HTOBE64(~(*(uint64_t *)val_1));
                atomic_64_masked_cs_mask_seg->compare = 0;

                if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI)
                    data_seg_ptr =
                        (struct mlx5_wqe_data_seg *)((uintptr_t)atomic_seg_2_ptr +
                                                     sizeof(*atomic_64_masked_cs_mask_seg));
                else
                    data_seg_ptr = (struct mlx5_wqe_data_seg *)out_wqes[1];
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (gic_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = 0;
                atomic_32_masked_cs_seg->compare_mask = 0;
                atomic_32_masked_cs_seg->swap_mask = HTOBE32(*(uint32_t *)val_1);
            } else {
                ++ds;
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                    (gic_atomic_64_masked_cs_seg_t *)&atomic_seg_1;
                atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_data_seg->compare = 0;

                gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                    (gic_atomic_64_masked_cs_seg_t *)&atomic_seg_2;
                atomic_64_masked_cs_mask_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_mask_seg->compare = 0;

                if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI)
                    data_seg_ptr =
                        (struct mlx5_wqe_data_seg *)((uintptr_t)atomic_seg_2_ptr +
                                                     sizeof(*atomic_64_masked_cs_mask_seg));
                else
                    data_seg_ptr = (struct mlx5_wqe_data_seg *)out_wqes[1];
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (gic_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_fa_seg->field_boundary = UINT32_MAX;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (gic_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_fa_seg->field_boundary = UINT64_MAX;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (gic_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = 0;
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_8_BYTE_EXT_AMO_OPMOD);

                gic_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (gic_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = 0;
                atomic_64_masked_fa_seg->field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (gic_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_FA | (wqe_idx << 8));
                atomic_seg_1.swap_add = HTOBE64(*(uint64_t *)val_1);
            }
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    GIC_4_BYTE_EXT_AMO_OPMOD);

                gic_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (gic_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = HTOBE32(*(uint32_t *)val_2);
                atomic_32_masked_cs_seg->compare_mask = UINT32_MAX;
                atomic_32_masked_cs_seg->swap_mask = UINT32_MAX;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_CS | (wqe_idx << 8));
                atomic_seg_1.swap_add = HTOBE64(*(uint64_t *)val_1);
                atomic_seg_1.compare = HTOBE64(*(uint64_t *)val_2);
            }
            break;
        }
        default: { assert(0); }
    }

    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    if (ds > 4) {
        ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
        *out_ctrl_seg = ctrl_seg;
    }

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) gic_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)atomic_seg_1_ptr;
    src = (uint32_t *)&atomic_seg_1;
    for (int i = 0; i < sizeof(*atomic_seg_1_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)atomic_seg_2_ptr;
    src = (uint32_t *)&atomic_seg_2;
    for (int i = 0; i < sizeof(*atomic_seg_2_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        gic_store_relaxed(&dst[i], src[i]);

    if (ds <= 4) {
        // Patch with NOP
        ++wqe_idx;

        ctrl_seg = {
            0,
        };
        ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | 1);
        ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
        ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_NOP);

        ctrl_seg_ptr = (gic_ctrl_seg_t *)out_wqes[1];

        dst = (uint32_t *)ctrl_seg_ptr;
        src = (uint32_t *)&ctrl_seg;
        for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
            gic_store_relaxed(&dst[i], src[i]);

        *out_ctrl_seg = ctrl_seg;
    }
}

__device__ static inline void gic_update_dbr(nvshmemi_gic_device_qp_t *qp, uint32_t dbrec_head) {
    // DBREC contains the index of the next empty WQEBB.
    __be32 dbrec_val;
    __be32 *dbrec_ptr = qp->tx_wq.dbrec;

    // This is equivalent to
    // WRITE_ONCE(dbrec_ptr, HTOBE32(dbrec_head & 0xffff));
    asm volatile(
        "{\n\t"
        ".reg .b32 mask1;\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 mask2;\n\t"
        "mov.b32 mask1, 0xffff;\n\t"
        "mov.b32 mask2, 0x123;\n\t"
        "and.b32 dbrec_head_16b, %1, mask1;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    gic_store_release(dbrec_ptr, dbrec_val);
}

__device__ static inline void gic_ring_db(nvshmemi_gic_device_qp_t *qp, gic_ctrl_seg_t *ctrl_seg) {
    uint64_t *bf_ptr = (uint64_t *)qp->tx_wq.bf;

    gic_store_release(bf_ptr, *(uint64_t *)ctrl_seg);
}

template <bool need_strong_flush>
__device__ static inline void gic_post_send(nvshmemi_gic_device_qp_t *qp, uint64_t new_cons_idx,
                                            gic_ctrl_seg_t *ctrl_seg) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_cons_idx;

    // Update cons_head before ringing the db so that we know which index is needed in quiet/fence.
    gic_lock_acquire<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);

    if (need_strong_flush)
        old_cons_idx = atomicMax((unsigned long long int *)&mvars->tx_wq.cons_head,
                                 (unsigned long long int)new_cons_idx);
    else
        old_cons_idx = atomicMax_block((unsigned long long int *)&mvars->tx_wq.cons_head,
                                       (unsigned long long int)new_cons_idx);

    if (likely(new_cons_idx > old_cons_idx)) {
        GIC_MEMBAR();
        gic_update_dbr(qp, new_cons_idx);
        GIC_MEMBAR();
        gic_ring_db(qp, ctrl_seg);
    }

    gic_lock_release<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);
}

// If `qp` is shared among CTAs, need_strong_flush must be set to true because
// we must push prior writes from this CTA to L2 before coalescing DB.
template <bool need_strong_flush>
__device__ static inline void gic_submit_requests(nvshmemi_gic_device_qp_t *qp,
                                                  uint64_t base_wqe_idx, uint16_t num_wqes,
                                                  gic_ctrl_seg_t *ctrl_seg) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint64_t mask = ~((uint64_t)(state->num_requests_in_batch - 1));

    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    // WQE writes must be finished first.
    if (need_strong_flush)
        // membar from a different CTA does not push prior writes of this CTA.
        // We must push them out first because a different CTA might post-send for us.
        GIC_MEMBAR_NO_OPTIMIZATION();
    else
        // It is ok for those wqes to not be visible to the GPU scope yet.
        // gic_post_send will take care of that (if we choose to call it).
        GIC_MFENCE();

    // Wait for prior WQE slots to be filled first.
    // They might not be post-sent yet.
    if (need_strong_flush)
        while (atomicCAS((unsigned long long int *)&mvars->tx_wq.wqe_tail,
                         (unsigned long long int)base_wqe_idx,
                         (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here
    else
        while (atomicCAS_block((unsigned long long int *)&mvars->tx_wq.wqe_tail,
                               (unsigned long long int)base_wqe_idx,
                               (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here

    GIC_MFENCE();

    bool do_post_send =
        (new_wqe_idx == gic_atomic_read(&mvars->tx_wq.wqe_head))  // No concurrent submissions
        || ((base_wqe_idx & mask) !=
            (new_wqe_idx & mask))  // Num of not-yet-posted wqes is beyond the threshold.
        || (num_wqes >= state->num_requests_in_batch);  // The number of wqes in this submission
                                                        // reaches the threshold.

    if (do_post_send) gic_post_send<need_strong_flush>(qp, new_wqe_idx, ctrl_seg);
}

template <bool support_half_av_seg>
__device__ static inline uint64_t gic_quiet(nvshmemi_gic_device_qp_t *qp) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint64_t wqe_tail = gic_atomic_read(&mvars->tx_wq.wqe_tail);
    nvshmemi_gic_device_cq_t cq = *qp->tx_wq.cq;

    int err = 0;
    int status = gic_poll_cq<support_half_av_seg>(&cq, wqe_tail, &err);
    // TODO: Integrate the error handler with the core NVSHMEM
#ifdef NVSHMEM_IBGDA_DEBUG
    if (status) {
        printf("gic_poll_cq failed with error=%d.\n", err);
    }
#endif
    assert(likely(status == 0));
    return wqe_tail;
}

template <bool support_half_av_seg>
__device__ static inline void gic_wait_for_slot_availability(nvshmemi_gic_device_qp_t *qp,
                                                             uint64_t wqe_idx) {
    // Don't have to use READ_ONCE here.
    // Even if we get stale cons_tail, poll_cq will catch that.
    int status = 0;
    int err = 0;
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint64_t cons_tail = mvars->tx_wq.cons_tail;
    uint16_t nwqes = qp->tx_wq.nwqes;

    assert(likely(wqe_idx >= cons_tail));
    if (unlikely(wqe_idx - cons_tail > nwqes)) {
        nvshmemi_gic_device_cq_t cq = *qp->tx_wq.cq;
        status = gic_poll_cq<support_half_av_seg>(&cq, wqe_idx - nwqes, &err);
        // TODO: Integrate the error handler with the core NVSHMEM
#ifdef NVSHMEM_IBGDA_DEBUG
        if (status) {
            printf("gic_poll_cq failed with error=%d.\n", err);
        }
#endif
        assert(likely(status == 0));
    }
    GIC_MFENCE();
}

__device__ static inline uint32_t gic_get_dct_id(int pe) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint32_t id = gic_get_ctaid();
    return (pe * state->ndcts_per_pe) + (id % state->ndcts_per_pe);
}

__device__ static inline nvshmemi_gic_device_dct_t *gic_get_dct(int pe) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint32_t dct_idx = gic_get_dct_id(pe);

    if (dct_idx < NVSHMEMI_GIC_MAX_CONST_DCTS) return &state->constmem.dcts[dct_idx];

    return &state->globalmem.dcts[dct_idx - NVSHMEMI_GIC_MAX_CONST_DCTS];
}

__device__ static inline nvshmemi_gic_device_qp_t *gic_get_dci(int pe,
                                                               bool *out_shared_among_ctas) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint32_t id;
    bool shared_among_ctas = false;
    uint32_t warpid = nvshmemi_thread_id_in_block() / nvshmemi_warp_size();
    switch (state->dci_map_type) {
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_CTA:
            id = gic_get_ctaid();
            break;
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_SM:
            id = gic_get_smid();
            shared_among_ctas = true;
            break;
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_WARP:
            id = gic_get_ctaid() * nvshmemi_block_size() + warpid;
            break;
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_DCT: {
            uint32_t dct_id = gic_get_dct_id(pe);
            uint32_t group_id = gic_get_ctaid() * nvshmemi_block_size() + warpid;
            id = (group_id % state->num_dct_groups) * state->ndcts_per_pe *
                     nvshmemi_device_state_d.npes +
                 dct_id;
            shared_among_ctas = true;
            break;
        }
        default:
            assert(0);
            break;
    }

    uint32_t idx;
    if (id < state->num_exclusive_dcis)
        idx = id;
    else {
        idx = state->num_exclusive_dcis + (id % state->num_shared_dcis);
        shared_among_ctas = true;
    }

    *out_shared_among_ctas = shared_among_ctas;
    return &state->globalmem.dcis[idx];
}

__device__ static inline nvshmemi_gic_device_qp_t *gic_get_rc(int pe, bool *out_shared_among_ctas) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint32_t id;
    uint32_t idx;
    uint32_t warpid = nvshmemi_thread_id_in_block() / nvshmemi_warp_size();

    assert(pe != nvshmemi_device_state_d.mype);

    switch (state->rc_map_type) {
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_CTA:
            id = gic_get_ctaid();
            break;
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_SM:
            id = gic_get_smid();
            break;
        case NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_WARP:
            id = gic_get_ctaid() * nvshmemi_block_size() + warpid;
            break;
        default:
            assert(0);
            break;
    }

    idx = (pe * state->num_rc_per_pe) + (id % state->num_rc_per_pe);

    *out_shared_among_ctas = true;
    return &state->globalmem.rcs[idx];
}

__device__ static inline nvshmemi_gic_device_qp_t *gic_get_qp(int pe, bool *out_shared_among_ctas) {
    nvshmemi_gic_device_state_t *state = gic_get_state();

    if (gic_is_rc_enabled() && pe != nvshmemi_device_state_d.mype)
        return gic_get_rc(pe, out_shared_among_ctas);
    else
        return gic_get_dci(pe, out_shared_among_ctas);
}

__device__ static inline void gic_get_lkey(uint64_t addr, __be32 *lkey, size_t *chunk_size) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t heap_end = heap_start + nvshmemi_device_state_d.heap_size - 1;
    if (heap_start <= addr && addr <= heap_end) {
        // addr in the symmetric heap
        uint64_t idx = (addr - heap_start) >> state->log2_cumem_granularity;
        nvshmemi_gic_device_key_t device_key;

        if (idx < NVSHMEMI_GIC_MAX_CONST_LKEYS)
            device_key = state->constmem.lkeys[idx];
        else
            device_key = state->globalmem.lkeys[idx - NVSHMEMI_GIC_MAX_CONST_LKEYS];

        assert(addr < device_key.next_addr);

        *lkey = device_key.key;
        *chunk_size = device_key.next_addr - addr;
        return;
    } else {
        // local-only addr
        nvshmemi_gic_device_local_only_mhandle_t *mhandle =
            state->globalmem.local_only_mhandle_head;

        while (mhandle) {
            if (mhandle->start <= addr && addr <= mhandle->end) {
                *lkey = mhandle->lkey;
                *chunk_size = mhandle->end - addr + 1;
                return;
            }
            mhandle = mhandle->next;
        }
    }

    // lkey is not found.
    assert(0);
}

__device__ static inline void gic_get_raddr_rkey(uint64_t addr, int pe, uint64_t *raddr,
                                                 __be32 *rkey, size_t *chunk_size) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t roffset = addr - heap_start;
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes) + pe;
    nvshmemi_gic_device_key_t device_key;

    if (idx < NVSHMEMI_GIC_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_GIC_MAX_CONST_RKEYS];

    assert(roffset < device_key.next_addr);

    *raddr = (uint64_t)nvshmemi_device_state_d.peer_heap_base_actual[pe] + roffset;
    *rkey = device_key.key;
    *chunk_size = device_key.next_addr - roffset;
}

template <bool support_half_av_seg>
__device__ static inline uint64_t gic_reserve_wqe_slots(nvshmemi_gic_device_qp_t *qp,
                                                        unsigned long long int num_wqes,
                                                        bool is_qp_shared_among_ctas) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint64_t wqe_idx;
#if CUDA_VERSION >= 12000
    if (is_qp_shared_among_ctas)
        wqe_idx = atomicAdd((unsigned long long int *)&mvars->tx_wq.wqe_head, num_wqes);
    else
        wqe_idx = atomicAdd_block((unsigned long long int *)&mvars->tx_wq.wqe_head, num_wqes);
#else
    // WAR NVBUG 3749055. The fix is in nvcc of CUDA 12.0 and later.
    if (is_qp_shared_among_ctas)
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;"
                     : "=l"(wqe_idx)
                     : "l"(&mvars->tx_wq.wqe_head), "l"(num_wqes));
    else
        asm volatile("atom.relaxed.cta.global.add.u64 %0, [%1], %2;"
                     : "=l"(wqe_idx)
                     : "l"(&mvars->tx_wq.wqe_head), "l"(num_wqes));
#endif
    // If last slot is available, all prior slots are also available.
    gic_wait_for_slot_availability<support_half_av_seg>(qp, wqe_idx + num_wqes);
    return wqe_idx;
}

__device__ static inline uint64_t gic_reserve_ibuf_slots(nvshmemi_gic_device_qp_t *qp,
                                                         unsigned long long int num_slots) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint32_t nslots = qp->ibuf.nslots;
    uint64_t base_idx = atomicAdd((unsigned long long int *)&mvars->ibuf.head, num_slots);
    uint64_t idx = base_idx + num_slots;

    // Wait until the slots become available.
    while (idx - gic_atomic_read(&mvars->ibuf.tail) > nslots)
        ;

    // Prevent the reordering of the above wait loop.
    GIC_MFENCE();

    return base_idx;
}

__device__ static inline void gic_release_ibuf(nvshmemi_gic_device_qp_t *qp,
                                               unsigned long long int base_idx,
                                               unsigned long long int num_slots) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    unsigned long long int new_idx = base_idx + num_slots;
    GIC_MFENCE();
    // Wait here.
    while (atomicCAS((unsigned long long int *)&mvars->ibuf.tail, (unsigned long long int)base_idx,
                     new_idx) != base_idx)
        ;
    GIC_MFENCE();
}

__device__ static inline uint64_t gic_get_ibuf_addr(nvshmemi_gic_device_qp_t *qp, uint64_t idx) {
    idx = idx & (qp->ibuf.nslots - 1);

    // buf[0] is reserved for non-fetch operations
    return (uint64_t)qp->ibuf.buf + NVSHMEMI_GIC_IBUF_SLOT_SIZE * (idx + 1);
}

__device__ static inline bool gic_can_coalesce_warp(unsigned int amask,
                                                    nvshmemi_gic_device_qp_t *qp) {
    int pred_same_qp;

    if (amask != GIC_FULL_WARP) return false;

    __match_all_sync(amask, qp->qpn, &pred_same_qp);
    if (!pred_same_qp) return false;

    return true;
}

template <bool support_half_av_seg>
__device__ static inline uint64_t gic_cst(nvshmemi_gic_device_qp_t *dci,
                                          bool is_dci_shared_among_ctas) {
    assert(likely(dci->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI));

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(nvshmemi_device_state_d.mype);

    uint64_t laddr = (uint64_t)dci->ibuf.buf;
    __be32 lkey = dci->ibuf.lkey;

    const int num_wqes = 1;

    uint64_t base_wqe_idx =
        gic_reserve_wqe_slots<support_half_av_seg>(dci, num_wqes, is_dci_shared_among_ctas);

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[1];
    wqe_ptrs[0] = gic_get_wqe_ptr(dci, base_wqe_idx);

    // DUMP OP causes the NIC to read laddr, which is always on GPU memory.
    // For CST, it is cheaper than RDMA READ.
    gic_write_dump_wqe(dci, laddr, lkey, sizeof(char), base_wqe_idx, GIC_MLX5_FM_NO_FENCE, wqe_ptrs,
                       &ctrl_seg);

    // Don't update get_head here because this is internal cst
    if (is_dci_shared_among_ctas)
        gic_submit_requests<true>(dci, base_wqe_idx, num_wqes, &ctrl_seg);
    else
        gic_submit_requests<false>(dci, base_wqe_idx, num_wqes, &ctrl_seg);

    return gic_quiet<support_half_av_seg>(dci);
}

template <bool support_half_av_seg>
__device__ static inline uint64_t gic_quiet_with_cst(nvshmemi_gic_device_qp_t *qp,
                                                     bool is_qp_shared_among_ctas) {
    nvshmemi_gic_device_qp_management_t *mvars = &qp->mvars;
    uint64_t get_head = gic_atomic_read(&mvars->tx_wq.get_head);
    uint64_t ticket = gic_quiet<support_half_av_seg>(qp);
    uint64_t get_tail = gic_atomic_read(&mvars->tx_wq.get_tail);

    // TODO: Change to WAIT + DUMP
    // In that case, we don't have to do quiet first
    if (get_tail < get_head) {
        if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) {
            ticket = gic_cst<support_half_av_seg>(qp, is_qp_shared_among_ctas);
            gic_update_get_tail(qp, ticket);
        } else {
            // We don't have RC loopback to self.
            // So, we grab a DCI for CST.
            bool is_dci_shared_among_ctas;
            nvshmemi_gic_device_qp_t *dci =
                gic_get_dci(nvshmemi_device_state_d.mype, &is_dci_shared_among_ctas);
            uint64_t cst_ticket = gic_cst<support_half_av_seg>(dci, is_dci_shared_among_ctas);
            gic_update_get_tail(dci, cst_ticket);
            gic_update_get_tail(qp, ticket);
        }
    }

    return ticket;
}

template <nvshmemi_op_t channel_op, bool nbi, bool support_half_av_seg>
__device__ static inline void gic_rma_thread(uint64_t rptr, uint64_t lptr, size_t remaining_size,
                                             int pe) {
    constexpr bool need_cst = (channel_op == NVSHMEMI_OP_GET);
    constexpr bool need_immediate_cst = !nbi && need_cst;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);

    int num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    bool did_quiet = false;

    if (unlikely(remaining_size == 0)) return;

    while (remaining_size > 0) {
        unsigned int amask = __activemask();

        int my_tid;
        int tg_size;

        __be32 lkey;
        size_t lchunk_size;
        gic_get_lkey(lptr, &lkey, &lchunk_size);

        __be32 rkey;
        uint64_t raddr;
        size_t rchunk_size;
        gic_get_raddr_rkey(rptr, pe, &raddr, &rkey, &rchunk_size);

        size_t transfer_size = gic_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);

        bool can_coalesce_warp = gic_can_coalesce_warp(amask, qp);

        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_wqes = num_wqes_per_cmd * tg_size + (need_immediate_cst ? 1 : 0);

        uint64_t base_wqe_idx;

        if (my_tid == 0) {
            base_wqe_idx =
                gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        }

        uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

        gic_ctrl_seg_t ctrl_seg;

        void *wqe_ptrs[2];
        wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                gic_write_rdma_write_wqe<support_half_av_seg>(qp, dct, lptr, lkey, raddr, rkey,
                                                              transfer_size, my_wqe_idx, wqe_ptrs,
                                                              &ctrl_seg);
                break;
            case NVSHMEMI_OP_GET:
                gic_write_rdma_read_wqe<support_half_av_seg>(qp, dct, lptr, lkey, raddr, rkey,
                                                             transfer_size, my_wqe_idx, wqe_ptrs,
                                                             &ctrl_seg);
                break;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported channel_op.\n");
#endif
                assert(0);
        }

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (need_immediate_cst) {
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
                gic_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                   my_wqe_idx, GIC_MLX5_FM_FENCE, wqe_ptrs, &ctrl_seg);
            } else if (need_cst) {
                // For nbi, we will do CST in QUIET.
                // GET index must be visible before the new cons index.
                gic_update_get_head(qp, base_wqe_idx + num_wqes);
            }

            if (is_qp_shared_among_ctas)
                gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
            else
                gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        }

        remaining_size -= transfer_size;

        rptr += transfer_size;
        lptr += transfer_size;

        if (can_coalesce_warp) {
            if (!nbi) {
                bool do_coalesce_quiet = __all_sync(amask, remaining_size == 0);
                if (do_coalesce_quiet && my_tid == tg_size - 1) {
                    // CST, if required, has already been enqueued. We simply need to
                    // do gic_quiet here.
                    gic_quiet<support_half_av_seg>(qp);
                }
                did_quiet |= do_coalesce_quiet;
            }
            nvshmemi_warp_sync();
        }
    }

    if (!nbi && !did_quiet) {
        // CST, if required, has already been enqueued. We simply need to
        // do gic_quiet here.
        gic_quiet<support_half_av_seg>(qp);
    }
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op, bool nbi, bool support_half_av_seg>
__device__ static inline void gic_rma(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int pe) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    constexpr bool need_cst = (channel_op == NVSHMEMI_OP_GET);
    constexpr bool need_immediate_cst = !nbi && need_cst;

    // Use only wrap 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();

    nvshmemi_gic_device_state_t *state = gic_get_state();

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    int num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    int num_wqes;

    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[2];

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = req_rptr;
    uint64_t lptr = req_lptr;

    __be32 lkey;
    __be32 my_lkey;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    if (unlikely(remaining_size == 0)) goto out;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        gic_get_lkey(lptr, &lkey, &lchunk_size);
        gic_get_raddr_rkey(rptr, pe, &raddr, &rkey, &rchunk_size);
        transfer_size = gic_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use gic_rma_thread to handle it instead.
    if (unlikely(chunk_idx > tg_size)) {
        if (my_tid == 0) {
            gic_rma_thread<channel_op, nbi, support_half_av_seg>(req_rptr, req_lptr, bytes, pe);
        }
        goto out;
    }

    num_wqes = num_wqes_per_cmd * chunk_idx + (need_immediate_cst ? 1 : 0);

    if (my_tid == 0) {
        base_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
    }

    base_wqe_idx = __shfl_sync(GIC_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    if (my_tid < chunk_idx) {
        wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                gic_write_rdma_write_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey, my_raddr,
                                                              my_rkey, my_transfer_size, my_wqe_idx,
                                                              wqe_ptrs, &ctrl_seg);
                break;
            case NVSHMEMI_OP_GET:
                gic_write_rdma_read_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey, my_raddr,
                                                             my_rkey, my_transfer_size, my_wqe_idx,
                                                             wqe_ptrs, &ctrl_seg);
                break;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported channel_op.\n");
#endif
                assert(0);
        }
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx - 1) {
        if (need_immediate_cst) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            gic_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char), my_wqe_idx,
                               GIC_MLX5_FM_FENCE, wqe_ptrs, &ctrl_seg);
        } else if (need_cst) {
            // For nbi, we will do CST in QUIET.
            // GET index must be visible before the new cons index.
            // gic_submit_requests has fence, which guarantees the ordering.
            gic_update_get_head(qp, base_wqe_idx + num_wqes);
        }

        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);

        if (!nbi) {
            // CST, if required, has already been enqueued. We simply need to
            // do gic_quiet here.
            gic_quiet<support_half_av_seg>(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

/**
 * RMA P base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 64) failed");
#endif
template <typename T, bool is_full_warp, bool can_combine_data, bool support_half_av_seg>
__device__ static inline void nvshmemi_gic_rma_p_impl(const uint64_t raddr, const __be32 rkey,
                                                      const T value, int pe) {
    static_assert((can_combine_data && is_full_warp) || (!can_combine_data),
                  "can_combine_data check 1 failed.\n");
    static_assert((can_combine_data && support_half_av_seg) || (!can_combine_data),
                  "can_combine_data check 2 failed.\n");

    int my_tid;
    int tg_size;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    if (is_full_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    int num_wqes_per_cmd;
    int num_wqes;

    if (can_combine_data) {
        if (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_RC) {
            num_wqes_per_cmd =
                gic_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_GIC_DEVICE_QP_TYPE_RC>();
        } else if (sizeof(T) == 8) {
            num_wqes_per_cmd =
                2 *
                gic_get_num_wqes_in_inl_combine_warp<uint32_t, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>();
        } else {
            num_wqes_per_cmd =
                gic_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI>();
        }
        num_wqes = num_wqes_per_cmd;
    } else {
        num_wqes_per_cmd =
            (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;
        num_wqes = num_wqes_per_cmd * tg_size;
    }

    uint64_t base_wqe_idx;

    if (my_tid == 0) {
        base_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (is_full_warp) {
        base_wqe_idx = __shfl_sync(GIC_FULL_WARP, base_wqe_idx, 0);
    }

    uint64_t my_wqe_idx =
        can_combine_data ? base_wqe_idx : base_wqe_idx + (my_tid * num_wqes_per_cmd);

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        wqe_ptrs[i] = gic_get_wqe_ptr(qp, my_wqe_idx + i);
    }

    if (can_combine_data && sizeof(T) == 8 && qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI)
        gic_write_rdma_write_inl_wqe_combine_warp_for_dci_8B<T>(
            qp, dct, value, raddr, rkey, my_wqe_idx, my_tid, wqe_ptrs, &ctrl_seg);
    else if (can_combine_data)
        gic_write_rdma_write_inl_wqe_combine_warp<T>(qp, dct, value, raddr, rkey, my_wqe_idx,
                                                     my_tid, wqe_ptrs, &ctrl_seg);
    else
        gic_write_rdma_write_inl_wqe<support_half_av_seg>(qp, dct, &value, raddr, rkey, sizeof(T),
                                                          my_wqe_idx, wqe_ptrs, &ctrl_seg);

    if (is_full_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
    }

    if (is_full_warp) nvshmemi_warp_sync();
}

template <typename T>
__device__ inline void nvshmemi_gic_rma_p(void *rptr, const T value, int pe) {
    unsigned int amask = __activemask();
    bool can_combine_data = false;
    int pred_pe = 0;
    int pred_contiguous = 0;
    int pred_rkey = 0;
    int my_tid;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    gic_get_raddr_rkey((uint64_t)rptr, pe, &raddr, &rkey, &rchunk_size);

    // With proper alignment (requirement of NVSHMEM), one element cannot span multiple chunks.
    assert(rchunk_size >= sizeof(T));

    if (amask == GIC_FULL_WARP) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        __match_all_sync(GIC_FULL_WARP, pe, &pred_pe);
        __match_all_sync(GIC_FULL_WARP, (uintptr_t)(rptr) - (my_tid * sizeof(T)), &pred_contiguous);
        __match_all_sync(GIC_FULL_WARP, rkey, &pred_rkey);
        can_combine_data = (pred_pe && pred_contiguous && pred_rkey && state->support_half_av_seg);
        if (can_combine_data)
            nvshmemi_gic_rma_p_impl<T, true, true, true>(raddr, rkey, value, pe);
        else if (state->support_half_av_seg)
            nvshmemi_gic_rma_p_impl<T, true, false, true>(raddr, rkey, value, pe);
        else
            nvshmemi_gic_rma_p_impl<T, true, false, false>(raddr, rkey, value, pe);
    } else if (state->support_half_av_seg)
        nvshmemi_gic_rma_p_impl<T, false, false, true>(raddr, rkey, value, pe);
    else
        nvshmemi_gic_rma_p_impl<T, false, false, false>(raddr, rkey, value, pe);
}

/**
 * RMA G base
 */
template <typename T, bool support_half_av_seg>
__device__ inline T nvshmemi_gic_rma_g_impl(void *rptr, int pe) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    T ret;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    gic_get_raddr_rkey((uint64_t)rptr, pe, &raddr, &rkey, &rchunk_size);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    bool can_coalesce_warp = gic_can_coalesce_warp(amask, qp);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    int num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    int num_wqes = num_wqes_per_cmd * tg_size + 1;

    if (my_tid == 0) {
        base_ibuf_idx = gic_reserve_ibuf_slots(qp, tg_size);
        base_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = base_ibuf_idx + my_tid;

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[2];
    wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

    uint64_t laddr = gic_get_ibuf_addr(qp, my_ibuf_idx);
    __be32 lkey = qp->ibuf.lkey;

    gic_write_rdma_read_wqe<support_half_av_seg>(qp, dct, laddr, lkey, raddr, rkey, sizeof(T),
                                                 my_wqe_idx, wqe_ptrs, &ctrl_seg);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        my_wqe_idx += num_wqes_per_cmd;
        wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
        // Enqueue CST op in the QP.  This command has NIC Fence, which
        // waits for all prior READ/ATOMIC to finish before issuing this
        // DUMP.
        gic_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char), my_wqe_idx,
                           GIC_MLX5_FM_FENCE, wqe_ptrs, &ctrl_seg);

        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);

        gic_quiet<support_half_av_seg>(qp);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) gic_release_ibuf(qp, base_ibuf_idx, tg_size);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    return ret;
}

template <typename T>
__device__ inline T nvshmemi_gic_rma_g(void *rptr, int pe) {
    T ret;
    nvshmemi_gic_device_state_t *state = gic_get_state();

    if (state->support_half_av_seg)
        ret = nvshmemi_gic_rma_g_impl<T, true>(rptr, pe);
    else
        ret = nvshmemi_gic_rma_g_impl<T, false>(rptr, pe);
    return ret;
}

/**
 * RMA NBI base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ inline void nvshmemi_gic_rma_nbi(void *rptr, void *lptr, size_t bytes, int pe) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (state->support_half_av_seg) {
            gic_rma_thread<channel_op, true, true>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        } else {
            gic_rma_thread<channel_op, true, false>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        }
    } else {
        if (state->support_half_av_seg) {
            gic_rma<SCOPE, channel_op, true, true>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        } else {
            gic_rma<SCOPE, channel_op, true, false>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        }
    }
}

/**
 * RMA (blocking) base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ inline void nvshmemi_gic_rma(void *rptr, void *lptr, size_t bytes, int pe) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (state->support_half_av_seg) {
            gic_rma_thread<channel_op, false, true>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        } else {
            gic_rma_thread<channel_op, false, false>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        }
    } else {
        if (state->support_half_av_seg) {
            gic_rma<SCOPE, channel_op, false, true>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        } else {
            gic_rma<SCOPE, channel_op, false, false>((uint64_t)rptr, (uint64_t)lptr, bytes, pe);
        }
    }
}

/**
 * AMO non-fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ inline void nvshmemi_gic_amo_nonfetch_impl(void *rptr, const T value, int pe,
                                                      nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    gic_get_raddr_rkey((uint64_t)rptr, pe, &raddr, &rkey, &rchunk_size);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    bool can_coalesce_warp = gic_can_coalesce_warp(amask, qp);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    int num_wqes_per_cmd = 2;
    int num_wqes = num_wqes_per_cmd * tg_size;

    uint64_t base_wqe_idx;

    if (my_tid == 0)
        base_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);

    if (can_coalesce_warp) base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[2];
    wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

    gic_write_atomic_wqe<support_half_av_seg>(qp, dct, &value, NULL, (uint64_t)qp->ibuf.buf,
                                              qp->ibuf.lkey, raddr, rkey, sizeof(T), my_wqe_idx, op,
                                              wqe_ptrs, &ctrl_seg);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();
}

template <typename T>
__device__ inline void nvshmemi_gic_amo_nonfetch(void *rptr, const T value, int pe,
                                                 nvshmemi_amo_t op) {
    nvshmemi_gic_device_state_t *state = gic_get_state();

    if (state->support_half_av_seg)
        nvshmemi_gic_amo_nonfetch_impl<T, true>(rptr, value, pe, op);
    else
        nvshmemi_gic_amo_nonfetch_impl<T, false>(rptr, value, pe, op);
}

/**
 * AMO fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ inline T nvshmemi_gic_amo_fetch_impl(void *rptr, const T value, const T compare, int pe,
                                                nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    T ret;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    gic_get_raddr_rkey((uint64_t)rptr, pe, &raddr, &rkey, &rchunk_size);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    bool can_coalesce_warp = gic_can_coalesce_warp(amask, qp);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    int num_wqes_per_cmd = 2;
    int num_wqes = num_wqes_per_cmd * tg_size + 1;

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    if (my_tid == 0) {
        base_ibuf_idx = gic_reserve_ibuf_slots(qp, tg_size);
        base_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = base_ibuf_idx + my_tid;

    uint64_t laddr = gic_get_ibuf_addr(qp, my_ibuf_idx);
    __be32 lkey = qp->ibuf.lkey;

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[2];
    wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

    gic_write_atomic_wqe<support_half_av_seg>(qp, dct, &value, &compare, laddr, lkey, raddr, rkey,
                                              sizeof(T), my_wqe_idx, op, wqe_ptrs, &ctrl_seg);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        my_wqe_idx += num_wqes_per_cmd;
        wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
        // Enqueue CST op in the QP.  This command has NIC Fence, which
        // waits for all prior READ/ATOMIC to finish before issuing this
        // DUMP.
        gic_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char), my_wqe_idx,
                           GIC_MLX5_FM_FENCE, wqe_ptrs, &ctrl_seg);

        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);

        gic_quiet<support_half_av_seg>(qp);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);
    if (sizeof(T) == 4) ret = BSWAP32((uint32_t)ret);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) gic_release_ibuf(qp, base_ibuf_idx, tg_size);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    return ret;
}

template <typename T>
__device__ inline T nvshmemi_gic_amo_fetch(void *rptr, const T value, const T compare, int pe,
                                           nvshmemi_amo_t op) {
    T ret;
    nvshmemi_gic_device_state_t *state = gic_get_state();

    if (state->support_half_av_seg)
        ret = nvshmemi_gic_amo_fetch_impl<T, true>(rptr, value, compare, pe, op);
    else
        ret = nvshmemi_gic_amo_fetch_impl<T, false>(rptr, value, compare, pe, op);
    return ret;
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 128,
              "static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 128) failed");
#endif
template <bool is_nbi, bool support_half_av_seg>
__device__ static inline void nvshmemi_gic_put_signal_thread_impl(void *rptr, void *lptr,
                                                                  size_t bytes, void *sig_rptr,
                                                                  uint64_t signal,
                                                                  nvshmemi_amo_t sig_op, int pe) {
    int my_tid;
    int tg_size;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    gic_ctrl_seg_t ctrl_seg;

    __be32 lkey;
    size_t lchunk_size;
    gic_get_lkey((uint64_t)lptr, &lkey, &lchunk_size);

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    gic_get_raddr_rkey((uint64_t)rptr, pe, &raddr, &rkey, &rchunk_size);

    __be32 sig_rkey;
    uint64_t sig_raddr;
    size_t sig_rchunk_size;
    gic_get_raddr_rkey((uint64_t)sig_rptr, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size);

    size_t transfer_size = gic_cal_transfer_size(bytes, lchunk_size, rchunk_size);

    if (transfer_size == bytes) {
        unsigned int amask = __activemask();

        bool can_coalesce_warp = gic_can_coalesce_warp(amask, qp);

        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_rdma_write_wqes_per_cmd =
            (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

        int num_atomic_wqes_per_cmd = 2;
        int num_wqes_per_cmd = num_rdma_write_wqes_per_cmd + num_atomic_wqes_per_cmd;
        int num_wqes = num_wqes_per_cmd * tg_size;

        uint64_t base_wqe_idx;

        if (my_tid == 0) {
            base_wqe_idx =
                gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        }

        uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

        void *wqe_ptrs[4];
        wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);
        wqe_ptrs[2] = gic_get_wqe_ptr(qp, my_wqe_idx + 2);
        wqe_ptrs[3] = gic_get_wqe_ptr(qp, my_wqe_idx + 3);

        gic_write_rdma_write_wqe<support_half_av_seg>(qp, dct, (uint64_t)lptr, lkey, raddr, rkey,
                                                      bytes, my_wqe_idx, wqe_ptrs, &ctrl_seg);

        gic_write_atomic_wqe<support_half_av_seg>(
            qp, dct, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_wqe_idx + num_rdma_write_wqes_per_cmd, sig_op,
            &wqe_ptrs[num_rdma_write_wqes_per_cmd], &ctrl_seg);

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (is_qp_shared_among_ctas)
                gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
            else
                gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);

            if (!is_nbi) {
                gic_quiet<support_half_av_seg>(qp);
            }
        }

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }
    } else {
        gic_rma_thread<NVSHMEMI_OP_PUT, true, support_half_av_seg>((uintptr_t)rptr, (uintptr_t)lptr,
                                                                   bytes, pe);

        uint64_t my_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, 2, is_qp_shared_among_ctas);

        void *wqe_ptrs[2];
        wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

        gic_write_atomic_wqe<support_half_av_seg>(
            qp, dct, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_wqe_idx, sig_op, wqe_ptrs, &ctrl_seg);

        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, my_wqe_idx, 2, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, my_wqe_idx, 2, &ctrl_seg);

        if (!is_nbi) {
            gic_quiet<support_half_av_seg>(qp);
        }
    }
}

/**
 * PUT SIGNAL base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GIC_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, bool is_nbi, bool support_half_av_seg>
__device__ static inline void nvshmemi_gic_put_signal_impl(void *req_rptr, void *req_lptr,
                                                           size_t bytes, void *sig_rptr,
                                                           uint64_t signal, nvshmemi_amo_t sig_op,
                                                           int pe) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    // Use only wrap 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();

    nvshmemi_gic_device_state_t *state = gic_get_state();

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(pe);
    bool is_qp_shared_among_ctas;
    nvshmemi_gic_device_qp_t *qp = gic_get_qp(pe, &is_qp_shared_among_ctas);

    int num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    int num_wqes;

    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    gic_ctrl_seg_t ctrl_seg;

    void *wqe_ptrs[2];

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = (uint64_t)req_rptr;
    uint64_t lptr = (uint64_t)req_lptr;

    __be32 lkey;
    __be32 my_lkey;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        gic_get_lkey(lptr, &lkey, &lchunk_size);
        gic_get_raddr_rkey(rptr, pe, &raddr, &rkey, &rchunk_size);
        transfer_size = gic_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use nvshmemi_gic_put_signal_thread_impl to handle it instead.
    // Note that we need one thread to handle amo.
    if (unlikely(chunk_idx > tg_size - 1)) {
        if (my_tid == 0) {
            nvshmemi_gic_put_signal_thread_impl<is_nbi, support_half_av_seg>(
                req_rptr, req_lptr, bytes, sig_rptr, signal, sig_op, pe);
        }
        goto out;
    }

    num_wqes = num_wqes_per_cmd * chunk_idx + 2;

    if (my_tid == 0) {
        base_wqe_idx =
            gic_reserve_wqe_slots<support_half_av_seg>(qp, num_wqes, is_qp_shared_among_ctas);
    }

    base_wqe_idx = __shfl_sync(GIC_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    wqe_ptrs[0] = gic_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = gic_get_wqe_ptr(qp, my_wqe_idx + 1);

    if (my_tid < chunk_idx) {
        gic_write_rdma_write_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey, my_raddr, my_rkey,
                                                      my_transfer_size, my_wqe_idx, wqe_ptrs,
                                                      &ctrl_seg);
    } else if (my_tid == chunk_idx) {
        __be32 sig_rkey;
        uint64_t sig_raddr;
        size_t sig_rchunk_size;
        gic_get_raddr_rkey((uint64_t)sig_rptr, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size);

        gic_write_atomic_wqe<support_half_av_seg>(
            qp, dct, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_wqe_idx, sig_op, wqe_ptrs, &ctrl_seg);
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx) {
        if (is_qp_shared_among_ctas)
            gic_submit_requests<true>(qp, base_wqe_idx, num_wqes, &ctrl_seg);
        else
            gic_submit_requests<false>(qp, base_wqe_idx, num_wqes, &ctrl_seg);

        if (!is_nbi) {
            gic_quiet<support_half_av_seg>(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_gic_put_signal(void *rptr, void *lptr, size_t bytes, void *sig_rptr,
                                               uint64_t signal, nvshmemi_amo_t sig_op, int pe,
                                               bool is_nbi) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (is_nbi && state->support_half_av_seg)
            nvshmemi_gic_put_signal_thread_impl<true, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                            sig_op, pe);
        else if (is_nbi && !state->support_half_av_seg)
            nvshmemi_gic_put_signal_thread_impl<true, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                             sig_op, pe);
        else if (!is_nbi && state->support_half_av_seg)
            nvshmemi_gic_put_signal_thread_impl<false, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                             sig_op, pe);
        else
            nvshmemi_gic_put_signal_thread_impl<false, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                              sig_op, pe);
    } else {
        if (is_nbi && state->support_half_av_seg)
            nvshmemi_gic_put_signal_impl<SCOPE, true, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                            sig_op, pe);
        else if (is_nbi && !state->support_half_av_seg)
            nvshmemi_gic_put_signal_impl<SCOPE, true, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                             sig_op, pe);
        else if (!is_nbi && state->support_half_av_seg)
            nvshmemi_gic_put_signal_impl<SCOPE, false, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                             sig_op, pe);
        else
            nvshmemi_gic_put_signal_impl<SCOPE, false, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                              sig_op, pe);
    }
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_gic_quiet() {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    nvshmemi_gic_device_qp_t *qp;
    uint32_t ndcis = state->num_shared_dcis + state->num_exclusive_dcis;
    uint32_t nrcs = state->num_rc_per_pe * nvshmemi_device_state_d.npes;
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();

    scope_size = scope_size > GIC_MAX_THREADS_PER_QUIET ? GIC_MAX_THREADS_PER_QUIET : scope_size;

    if (index_in_scope < scope_size) {
        for (uint32_t i = index_in_scope; i < ndcis; i += scope_size) {
            qp = &state->globalmem.dcis[i];
            if (state->support_half_av_seg)
                gic_quiet_with_cst<true>(qp, true);
            else
                gic_quiet_with_cst<false>(qp, true);
        }

        for (uint32_t i = index_in_scope; i < nrcs; i += scope_size) {
            if (i / state->num_rc_per_pe == nvshmemi_device_state_d.mype) continue;

            qp = &state->globalmem.rcs[i];
            gic_quiet_with_cst<true>(qp, true);
        }
    }
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_gic_fence() {
    // Multiple QPs may target the same PE before fence.
    // We need to quiet those QPs.
    // TODO: Make it more efficient.
    nvshmemi_gic_device_state_t *state = gic_get_state();
    uint32_t ndcis = state->num_shared_dcis + state->num_exclusive_dcis;
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();
    uint32_t nrcs = state->num_rc_per_pe * nvshmemi_device_state_d.npes;
    nvshmemi_gic_device_qp_t *qp;

    // As all WQEs always go to the same QP, FENCE is naturally guaranteed.
    if (unlikely(ndcis + nrcs <= 1)) return;

    scope_size = scope_size > GIC_MAX_THREADS_PER_QUIET ? GIC_MAX_THREADS_PER_QUIET : scope_size;

    // Fence does not guarantee the completion of prior operations.
    // It is ok for GET to finish without data arrival.
    // Use gic_quiet here instead of gic_quiet_with_cst since it is cheaper.
    if (index_in_scope < scope_size) {
        if (state->support_half_av_seg) {
            for (uint32_t i = index_in_scope; i < ndcis; i += scope_size) {
                qp = &state->globalmem.dcis[i];
                gic_quiet<true>(qp);
            }
        } else {
            for (uint32_t i = nvshmemi_thread_id_in_warp(); i < ndcis; i += warpSize) {
                qp = &state->globalmem.dcis[i];
                gic_quiet<false>(qp);
            }
        }

        for (uint32_t i = index_in_scope; i < nrcs; i += scope_size) {
            if (i / state->num_rc_per_pe == nvshmemi_device_state_d.mype) continue;
            qp = &state->globalmem.rcs[i];
            gic_quiet<true>(qp);
        }
    }

    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ inline void nvshmemi_gic_enforce_consistency_at_target(bool use_membar) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    bool is_dci_shared_among_ctas;
    // We don't have RC loopback to self.
    // So, DCI is always used here.
    nvshmemi_gic_device_qp_t *dci =
        gic_get_dci(nvshmemi_device_state_d.mype, &is_dci_shared_among_ctas);

    if (state->support_half_av_seg)
        gic_cst<true>(dci, is_dci_shared_among_ctas);
    else
        gic_cst<false>(dci, is_dci_shared_among_ctas);

    // TODO: This fence is from the design of Proxy.
    // Review if we still need it when we fully move to GIC -- especially for on-stream API.
    if (use_membar) {
        __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                                 // cst_ack_d (breaks cst -> rma)
    }
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_GIC_DEVICE_H_ */
