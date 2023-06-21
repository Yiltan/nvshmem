/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"

#define DEFN_NVSHMEMX_TYPENAME_FCOLLECT_ON_STREAM(TYPENAME, TYPE)                                  \
    int nvshmemx_##TYPENAME##_fcollect_on_stream(                                                  \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, cudaStream_t stream) { \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                            \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        return nvshmemi_fcollect_on_stream<TYPE>(team, dest, source, nelems, stream);              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_ON_STREAM)
