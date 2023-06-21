/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_IBGDA_H_
#define _NVSHMEMI_IBGDA_H_

#include <linux/types.h>
#include <stdint.h>
#include <infiniband/mlx5dv.h>

#define NVSHMEMI_GIC_MIN_QP_DEPTH 128
#define NVSHMEMI_GIC_MAX_QP_DEPTH 32768

#define NVSHMEMI_GIC_CQE_SIZE 64
#define NVSHMEMI_GIC_MAX_INLINE_SIZE (8 * 32)
#define NVSHMEMI_GIC_IBUF_SLOT_SIZE 8

#define NVSHMEMI_GIC_MAX_CONST_LKEYS 64
#define NVSHMEMI_GIC_MAX_CONST_RKEYS 64
#define NVSHMEMI_GIC_MAX_CONST_DCTS 128

/* These values are not defined on all systems.
 * However, they can be traced back to a kernel enum with
 * these values.
 */
#ifndef MLX5DV_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_BF 0x0
#endif

#ifndef MLX5DV_UAR_ALLOC_TYPE_NC
#define MLX5DV_UAR_ALLOC_TYPE_NC 0x1
#endif

typedef enum {
    NVSHMEMI_GIC_DEVICE_QP_TYPE_DCI = 1,
    NVSHMEMI_GIC_DEVICE_QP_TYPE_DCT = 2,
    NVSHMEMI_GIC_DEVICE_QP_TYPE_RC = 3
} nvshmemi_gic_device_qp_type_t;

typedef enum {
    NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_CTA = 0,
    NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_SM,
    NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_WARP,
    NVSHMEMI_GIC_DEVICE_QP_MAP_TYPE_DCT
} nvshmemi_gic_device_qp_map_type_t;

typedef struct {
    void *cqe;
    uint64_t *cons_head;
    uint64_t *cons_tail;
    uint64_t *wqe_head;
    uint64_t *wqe_tail;
    uint32_t cqn;
    uint32_t ncqes;
    uint32_t qpn;
    nvshmemi_gic_device_qp_type_t qp_type;
    __be32 *dbrec;
} nvshmemi_gic_device_cq_t;

// Variables for queue management.
// They are always in global memory.
typedef struct {
    int post_send_lock;
    struct {
        // All indexes are in wqebb unit
        uint64_t wqe_head;   // next not-yet-reserved wqe idx
        uint64_t wqe_tail;   // next not-yet-posted-but-submitted wqe idx
        uint64_t cons_head;  // num wqes that have been posted
        uint64_t cons_tail;  // num wqes that have been polled
        uint64_t get_head;   // last wqe idx + 1 with a "fetch" operation (g, get, amo_fetch)
        uint64_t get_tail;   // last wqe idx + 1 polled with cst; get_tail > get_head is possible
    } tx_wq;
    struct {
        uint64_t head;
        uint64_t tail;
    } ibuf;
} __attribute__((__aligned__(8))) nvshmemi_gic_device_qp_management_t;

typedef struct nvshmemi_gic_device_qp {
    nvshmemi_gic_device_qp_type_t qp_type;
    uint32_t qpn;
    struct {
        uint32_t nslots;  // num slots for fetch; always a power of 2
        void *buf;        // first NVSHMEMI_GIC_IBUF_SLOT_SIZE is for non-fetch
        __be32 lkey;
        __be32 rkey;
    } ibuf;  // Internal buffer
    struct {
        uint16_t nwqes;  // num wqes; some wqes may consume n wqebbs
        void *wqe;
        __be32 *dbrec;
        void *bf;
        nvshmemi_gic_device_cq_t *cq;
    } tx_wq;
    nvshmemi_gic_device_qp_management_t mvars;  // management variables
} nvshmemi_gic_device_qp_t;

typedef struct mlx5_wqe_av nvshmemi_gic_device_dct_t;

typedef struct nvshmemi_gic_device_local_only_mhandle {
    __be32 lkey;
    uint64_t start;
    uint64_t end;
    struct nvshmemi_gic_device_local_only_mhandle *next;
} nvshmemi_gic_device_local_only_mhandle_t;

typedef struct {
    __be32 key;
    uint64_t next_addr;  // end of this address range + 1
} nvshmemi_gic_device_key_t;

typedef struct {
    size_t log2_cumem_granularity;
    uint32_t num_shared_dcis;
    uint32_t num_exclusive_dcis;
    nvshmemi_gic_device_qp_map_type_t dci_map_type;
    uint32_t ndcts_per_pe;
    uint32_t num_dct_groups;
    uint32_t num_rc_per_pe;
    nvshmemi_gic_device_qp_map_type_t rc_map_type;
    uint32_t num_requests_in_batch; /* always a power of 2 */
    bool nic_buf_on_gpumem;
    bool support_half_av_seg;

    struct {
        nvshmemi_gic_device_cq_t *cqs;  // For both dcis and rcs. CQs for DCIs come first.
        nvshmemi_gic_device_qp_t *dcis;
        nvshmemi_gic_device_qp_t *rcs;
        nvshmemi_gic_device_local_only_mhandle *local_only_mhandle_head;

        // For dcts that cannot be contained in constmem.lkeys.
        // dcts[idx - NVSHMEMI_GIC_MAX_CONST_DCTS] gives the dct of idx.
        nvshmemi_gic_device_dct_t *dcts;

        // For lkeys that cannot be contained in constmem.lkeys.
        // lkeys[idx - NVSHMEMI_GIC_MAX_CONST_LKEYS] gives the lkey of chunk idx.
        nvshmemi_gic_device_key_t *lkeys;

        // For rkeys that cannot be contained in constmem.rkeys.
        // rkeys[(idx * npes + pe) - NVSHMEMI_GIC_MAX_CONST_RKEYS] gives rkey of chunck idx
        // targeting peer pe.
        nvshmemi_gic_device_key_t *rkeys;
    } globalmem;

    struct {
        // lkeys[idx] gives the lkey of chunk idx.
        nvshmemi_gic_device_key_t lkeys[NVSHMEMI_GIC_MAX_CONST_LKEYS];

        // rkeys[idx * npes + pe] gives rkey of chunck idx targeting peer pe.
        nvshmemi_gic_device_key_t rkeys[NVSHMEMI_GIC_MAX_CONST_RKEYS];

        nvshmemi_gic_device_dct_t dcts[NVSHMEMI_GIC_MAX_CONST_DCTS];
    } constmem;
} nvshmemi_gic_device_state_t;

#if defined(__CUDACC_RDC__)
#define EXTERN_CONSTANT extern __constant__
EXTERN_CONSTANT nvshmemi_gic_device_state_t nvshmemi_gic_device_state_d;
#undef EXTERN_CONSTANT
#endif

#endif /* _NVSHMEMI_GIC_H_ */
