/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __COMMON_H
#define __COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "nvshmemi_constants.h"

enum {
    NO_NBI = 0,
    NBI,
};

enum {
    NO_ASYNC = 0,
    ASYNC,
};

enum {
    SRC_STRIDE_CONTIG = 1,
};

enum {
    DEST_STRIDE_CONTIG = 1,
};

enum {
    UINT = 0,
    ULONG,
    ULONGLONG,
    INT32,
    INT64,
    UINT32,
    UINT64,
    INT,
    LONG,
    LONGLONG,
    SIZE,
    PTRDIFF,
    FLOAT,
    DOUBLE
};

#define NOT_A_CUDA_STREAM ((cudaStream_t)0)

#endif
