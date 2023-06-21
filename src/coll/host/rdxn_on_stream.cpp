/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM(TYPENAME, TYPE, OP)                     \
    int nvshmemx_##TYPENAME##_##OP##_reduce_on_stream(nvshmem_team_t team, TYPE *dest,     \
                                                      const TYPE *source, size_t nreduce,  \
                                                      cudaStream_t stream) {               \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                      \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                 \
        return nvshmemi_reduce_on_stream<TYPE, RDXN_OPS_##OP>(team, dest, source, nreduce, \
                                                              stream);                     \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, prod)
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM
