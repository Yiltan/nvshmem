/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY
#include "nvshmem_api.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"
#include "nvshmemx_error.h"

#define NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    void nvshmem_##Name##_atomic_inc(TYPE *target, int pe) {                        \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_inc() not implemented", \
                             nvshmemi_state->mype);                                 \
    }

NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    void nvshmem_##Name##_atomic_add(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_add() not implemented", \
                             nvshmemi_state->mype);                                 \
    }

NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    void nvshmem_##Name##_atomic_set(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_set() not implemented", \
                             nvshmemi_state->mype);                                 \
    }
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(longlong, LONGLONG, long long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(float, FLOAT, float)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(double, DOUBLE, double)

#define NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    void nvshmem_##Name##_atomic_and(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_and() not implemented", \
                             nvshmemi_state->mype);                                 \
    }
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    void nvshmem_##Name##_atomic_or(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_or() not implemented", \
                             nvshmemi_state->mype);                                \
    }
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    void nvshmem_##Name##_atomic_xor(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_xor() not implemented", \
                             nvshmemi_state->mype);                                 \
    }
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_fetch(const TYPE *target, int pe) {                  \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch() not implemented", \
                             nvshmemi_state->mype);                                   \
        return 0;                                                                     \
    }
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(longlong, LONGLONG, long long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(float, FLOAT, float)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(double, DOUBLE, double)

#define NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_fetch_inc(TYPE *target, int pe) {                        \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_inc() not implemented", \
                             nvshmemi_state->mype);                                       \
        return 0;                                                                         \
    }

NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                  \
    TYPE nvshmem_##Name##_atomic_fetch_add(TYPE *target, TYPE value, int pe) {       \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fadd() not implemented", \
                             nvshmemi_state->mype);                                  \
        return 0;                                                                    \
    }

NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_swap(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_swap() not implemented", \
                             nvshmemi_state->mype);                                  \
        return 0;                                                                    \
    }
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(longlong, LONGLONG, long long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(float, FLOAT, float)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(double, DOUBLE, double)

#define NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_compare_swap(TYPE *target, TYPE cond, TYPE value, int pe) { \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_compare_swap() not implemented", \
                             nvshmemi_state->mype);                                          \
        return value;                                                                        \
    }
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(int, INT, int)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(long, LONG, long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(longlong, LONGLONG, long long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(size, SIZE, size_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t)

#define NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_fetch_and(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_and() not implemented", \
                             nvshmemi_state->mype);                                       \
        return value;                                                                     \
    }
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_fetch_or(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_or() not implemented", \
                             nvshmemi_state->mype);                                      \
        return value;                                                                    \
    }
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                       \
    TYPE nvshmem_##Name##_atomic_fetch_xor(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_xor() not implemented", \
                             nvshmemi_state->mype);                                       \
        return value;                                                                     \
    }
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(uint64, UINT64, uint64_t)
