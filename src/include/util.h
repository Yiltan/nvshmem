/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include <stddef.h>
#include <sstream>
#include <vector>
#include <inttypes.h>
#include "nvshmem_build_options.h"
#include "nvshmemx_error.h"
#include "error_codes_internal.h"
#include "debug.h"
#include "env_defs.h"

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#define NZ_DEBUG_JMP(status, err, label, ...)                                               \
    do {                                                                                    \
        if (unlikely(status != 0)) {                                                        \
            if (nvshmem_debug_level >= NVSHMEM_LOG_TRACE) {                                 \
                fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
                fprintf(stderr, __VA_ARGS__);                                               \
            }                                                                               \
            status = err;                                                                   \
            goto label;                                                                     \
        }                                                                                   \
    } while (0)

#define CUDA_DRIVER_CHECK(cmd)                    \
    do {                                          \
        CUresult r = cmd;                         \
        cuGetErrorString(r, &p_err_str);          \
        if (unlikely(CUDA_SUCCESS != r)) {        \
            WARN("Cuda failure '%s'", p_err_str); \
            return NVSHMEMI_UNHANDLED_CUDA_ERROR; \
        }                                         \
    } while (false)

#define CUDA_CHECK(stmt)                                                                      \
    do {                                                                                      \
        CUresult result = (stmt);                                                             \
        cuGetErrorString(result, &p_err_str);                                                 \
        if (unlikely(CUDA_SUCCESS != result)) {                                               \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, p_err_str); \
            exit(-1);                                                                         \
        }                                                                                     \
        assert(CUDA_SUCCESS == result);                                                       \
    } while (0)

#define CUDA_RUNTIME_CHECK(stmt)                                                  \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#define CUDA_RUNTIME_CHECK_GOTO(stmt, res, label)                                 \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            res = NVSHMEMI_UNHANDLED_CUDA_ERROR;                                  \
            goto label;                                                           \
        }                                                                         \
    } while (0)

#define NCCL_CHECK(cmd)                                                   \
    do {                                                                  \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess) {                                           \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   nccl_ftable.GetErrorString(r));                        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

#define CUDA_DRIVER_ERROR_STRING(result)                                                      \
    do {                                                                                      \
        if (unlikely(CUDA_SUCCESS != result)) {                                               \
            cuGetErrorString(result, &p_err_str);                                             \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, p_err_str); \
        }                                                                                     \
    } while (0)

#define NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS()                                        \
    if (nvshmemi_is_limited_mpg_run) {                                                           \
        fprintf(stderr,                                                                          \
                "[%s:%d] Called NVSHMEM API not supported with limited MPG (Multiple Processes " \
                "Per GPU) runs\n",                                                               \
                __FILE__, __LINE__);                                                             \
        exit(-1);                                                                                \
    }

#define NVSHMEMU_THREAD_CS_INIT nvshmemu_thread_cs_init
#define NVSHMEMU_THREAD_CS_ENTER nvshmemu_thread_cs_enter
#define NVSHMEMU_THREAD_CS_EXIT nvshmemu_thread_cs_exit
#define NVSHMEMU_THREAD_CS_FINALIZE nvshmemu_thread_cs_finalize

#define NVSHMEMU_MAPPED_PTR_TRANSLATE(toPtr, fromPtr, peer)           \
    toPtr = (void *)((char *)(nvshmemi_state->peer_heap_base[peer]) + \
                     ((char *)fromPtr - (char *)(nvshmemi_state->heap_base)));

#define NVSHMEMU_UNMAPPED_PTR_TRANSLATE(toPtr, fromPtr, peer)                \
    toPtr = (void *)((char *)(nvshmemi_state->peer_heap_base_actual[peer]) + \
                     ((char *)fromPtr - (char *)(nvshmemi_state->heap_base)));

void nvshmemu_thread_cs_init();
void nvshmemu_thread_cs_finalize();
void nvshmemu_thread_cs_enter();
void nvshmemu_thread_cs_exit();

int nvshmemu_get_num_gpus_per_node();

uint64_t getHostHash();
nvshmemResult_t nvshmemu_gethostname(char *hostname, int maxlen);
void setup_sig_handler();
char *nvshmemu_hexdump(void *ptr, size_t len);
void nvshmemu_debug_log_cpuset(int category, const char *thread_name);

#define NVSHMEMI_WRAPLEN 80
char *nvshmemu_wrap(const char *str, const size_t wraplen, const char *indent,
                    const int strip_backticks);

extern const char *p_err_str;

extern struct nvshmemi_options_s nvshmemi_options;

enum { NVSHMEMI_OPTIONS_STYLE_INFO, NVSHMEMI_OPTIONS_STYLE_RST };

int nvshmemi_options_init(void);
void nvshmemi_options_print(int style);

#endif
