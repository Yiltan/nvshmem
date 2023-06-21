/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMI_CONSTANTS_H_
#define _NVSHMEMI_CONSTANTS_H_

#include "nvshmemi_transport_defines.h"

#define NVSHMEMI_DIAG_STRLEN 1024

#define SYNC_SIZE 27648 /*XXX:Number of GPUs on Summit; currently O(N), need to be O(1)*/
#define NVSHMEMI_SYNC_VALUE 0
#define NVSHMEMI_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEMI_BARRIER_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEMI_BCAST_SYNC_SIZE (10 * SYNC_SIZE)
#define NVSHMEMI_FCOLLECT_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_REDUCE_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE SYNC_SIZE
#define NVSHMEMI_COLLECT_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_ALLTOALL_SYNC_SIZE 1

#define NVSHMEMI_WARP_SIZE 32

typedef enum rdxn_ops {
    RDXN_OPS_AND = 0,
    RDXN_OPS_and = 0,
    RDXN_OPS_OR = 1,
    RDXN_OPS_or = 1,
    RDXN_OPS_XOR = 2,
    RDXN_OPS_xor = 2,
    RDXN_OPS_MIN = 3,
    RDXN_OPS_min = 3,
    RDXN_OPS_MAX = 4,
    RDXN_OPS_max = 4,
    RDXN_OPS_SUM = 5,
    RDXN_OPS_sum = 5,
    RDXN_OPS_PROD = 6,
    RDXN_OPS_prod = 6,
    RDXN_OPS_MAXLOC = 7,
    RDXN_OPS_maxloc
} rdxn_ops_t;

typedef enum {
    NVSHMEMI_JOB_GPU_LDST_ATOMICS = 1,
    NVSHMEMI_JOB_GPU_LDST = 1 << 1,
    NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS = 1 << 2,
    NVSHMEMI_JOB_GPU_PROXY = 1 << 3,
    NVSHMEMI_JOB_GPU_PROXY_CST = 1 << 4,
} nvshmemi_job_connectivity_t;

typedef enum {
    NVSHMEMI_PROXY_NONE = 0,
    NVSHMEMI_PROXY_MINIMAL = 1,
    NVSHMEMI_PROXY_FULL = 1 << 1,
} nvshmemi_proxy_status;

typedef struct {
    int major;
    int minor;
    int patch;
} nvshmemi_version_t;

typedef enum {
    NVSHMEMI_PE_DIST_ROUNDROBIN = 0,
    NVSHMEMI_PE_DIST_BLOCK,
    NVSHMEMI_PE_DIST_MISC
} nvshmemi_pe_dist_t;

#endif
