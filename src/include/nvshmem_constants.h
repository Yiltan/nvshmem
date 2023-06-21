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

#ifndef _NVSHMEM_CONSTANTS_H_
#define _NVSHMEM_CONSTANTS_H_

#include "nvshmemi_constants.h"
#include "nvshmem_version.h"

/* This is not the NVSHMEM release version, it is the supported OpenSHMEM spec version. */
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3

#define NVSHMEM_VENDOR_VERSION                                                   \
    ((NVSHMEM_VENDOR_MAJOR_VERSION)*10000 + (NVSHMEM_VENDOR_MINOR_VERSION)*100 + \
     (NVSHMEM_VENDOR_PATCH_VERSION))

#define NVSHMEMI_SUBST_AND_STRINGIFY_HELPER(S) #S
#define NVSHMEMI_SUBST_AND_STRINGIFY(S) NVSHMEMI_SUBST_AND_STRINGIFY_HELPER(S)

#define NVSHMEM_VENDOR_STRING \
    "NVSHMEM v"                                       \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_MAJOR_VERSION) "."      \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_MINOR_VERSION) "."      \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_PATCH_VERSION)

#define NVSHMEM_MAX_NAME_LEN 256

enum nvshmemi_cmp_type {
    NVSHMEM_CMP_EQ = 0,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE
};

enum nvshmemi_thread_support {
    NVSHMEM_THREAD_SINGLE = 0,
    NVSHMEM_THREAD_FUNNELED,
    NVSHMEM_THREAD_SERIALIZED,
    NVSHMEM_THREAD_MULTIPLE
};

enum nvshmem_signal_ops {
    NVSHMEM_SIGNAL_SET = NVSHMEMI_AMO_SIGNAL_SET,
    NVSHMEM_SIGNAL_ADD = NVSHMEMI_AMO_SIGNAL_ADD
};

enum {
    NVSHMEM_STATUS_NOT_INITIALIZED = 0,
    NVSHMEM_STATUS_IS_BOOTSTRAPPED,
    NVSHMEM_STATUS_IS_INITIALIZED,
    NVSHMEM_STATUS_LIMITED_MPG,
    NVSHMEM_STATUS_FULL_MPG
};

#endif
