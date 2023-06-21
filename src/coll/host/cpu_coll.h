/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_COLL_CPU_H
#define NVSHMEMI_COLL_CPU_H 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "util.h"
#include "nvshmem_api.h"
#include "nvshmemx_api.h"
#include "nvshmem_internal.h"

#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"

template <rdxn_ops_t op>
inline ncclRedOp_t nvshmemi_get_nccl_op();

template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_SUM>() {
    return ncclSum;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_PROD>() {
    return ncclProd;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_MIN>() {
    return ncclMin;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_MAX>() {
    return ncclMax;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_AND>() {
    return ncclNumOps;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_OR>() {
    return ncclNumOps;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_XOR>() {
    return ncclNumOps;
}

/* Reduction datatypes */
/*
 * ncclChar is an unsigned type. char in c++ can be signed or unsigned
 * so pick the "right" nccl type depending on the implementation of char.
 */
template <typename T>
inline ncclDataType_t nvshmemi_get_nccl_dt();

template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<char>() {
#if (CHAR_MIN == 0)
    return ncclUint8;
#else
    return ncclChar;
#endif
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<signed char>() {
    return ncclChar;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<short>() {
    return ncclNumTypes;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<int>() {
    return ncclInt;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<long>() {
    return ncclInt64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<long long>() {
    return ncclInt64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned char>() {
    return ncclUint8;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned short>() {
    return ncclNumTypes;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned int>() {
    return ncclUint32;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned long>() {
    return ncclUint64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned long long>() {
    return ncclUint64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<float>() {
    return ncclFloat;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<double>() {
    return ncclDouble;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<long double>() {
    return ncclNumTypes;
}
#ifdef NVSHMEM_COMPLEX_SUPPORT
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<complex double>() {
    return ncclNumTypes;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<complex float>() {
    return ncclNumTypes;
}
#endif

struct nccl_function_table {
    ncclResult_t (*GetVersion)(int* version);
    const char* (*GetErrorString)(ncclResult_t result);
    ncclResult_t (*GetUniqueId)(ncclUniqueId* uniqueId);
    ncclResult_t (*CommInitRank)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
    ncclResult_t (*CommDestroy)(ncclComm_t comm);
    ncclResult_t (*AllReduce)(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                              cudaStream_t stream);
    ncclResult_t (*Broadcast)(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, int root, ncclComm_t comm,
                              cudaStream_t stream);
    ncclResult_t (*AllGather)(const void* sendbuff, void* recvbuff, size_t sendcount,
                              ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t (*GroupStart)();
    ncclResult_t (*GroupEnd)();
    ncclResult_t (*Send)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
                         ncclComm_t comm, cudaStream_t stream);
    ncclResult_t (*Recv)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                         ncclComm_t comm, cudaStream_t stream);
};

extern struct nccl_function_table nccl_ftable;

#endif /* NVSHMEM_USE_NCCL */

#include "rdxn.h"
#include "alltoall.h"
#include "barrier.h"
#include "broadcast.h"
#include "fcollect.h"

/* macro definitions */
#define NVSHMEMI_COLL_CPU_STATUS_SUCCESS 0
#define NVSHMEMI_COLL_CPU_STATUS_ERROR 1

/* function declarations */
int nvshmemi_coll_common_cpu_read_env();
int nvshmemi_coll_common_cpu_init();
int nvshmemi_coll_common_cpu_finalize();

#define NVSHMEMI_COLL_CPU_ERR_POP()                                                         \
    do {                                                                                    \
        fprintf(stderr, "[pe = %d] Error at %s:%d in %s\n", nvshmemi_state->mype, __FILE__, \
                __LINE__, __FUNCTION__);                                                    \
        fflush(stderr);                                                                     \
        goto fn_fail;                                                                       \
    } while (0)

#endif /* NVSHMEMI_COLL_CPU_H */
