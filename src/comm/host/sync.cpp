/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"
#include "util.h"

#include "cuda_interface_sync.h"

#define NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE)                                     \
    void nvshmemx_##type##_wait_until_on_stream(TYPE *ivar, int cmp, TYPE cmp_value,       \
                                                cudaStream_t cstream) {                    \
        NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);                                          \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                 \
        call_nvshmemi_##type##_wait_until_on_stream_kernel(ivar, cmp, cmp_value, cstream); \
    }
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM

#define NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM(type, TYPE)                                         \
    void nvshmemx_##type##_wait_until_all_on_stream(TYPE *ivars, size_t nelems, const int *status, \
                                                    int cmp, TYPE cmp_value,                       \
                                                    cudaStream_t cstream) {                        \
        NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);                                                  \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        call_nvshmemi_##type##_wait_until_all_on_stream_kernel(ivars, nelems, status, cmp,         \
                                                               cmp_value, cstream);                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM

#define NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM(type, TYPE)                                 \
    void nvshmemx_##type##_wait_until_all_vector_on_stream(                                       \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value,                  \
        cudaStream_t cstream) {                                                                   \
        NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);                                                 \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                        \
        call_nvshmemi_##type##_wait_until_all_vector_on_stream_kernel(ivars, nelems, status, cmp, \
                                                                      cmp_value, cstream);        \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM

void nvshmemx_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value,
                                          cudaStream_t cstream) {
    NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    call_nvshmemi_signal_wait_until_on_stream_kernel(sig_addr, cmp, cmp_value, cstream);
}

void nvshmemi_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm) {
    int status = 0;
    if (sig_op == NVSHMEM_SIGNAL_SET && nvshmemi_state->peer_heap_base[pe] != NULL) {
        void *peer_addr;
        NVSHMEMU_MAPPED_PTR_TRANSLATE(peer_addr, sig_addr, pe)
        status = cudaMemcpyAsync(peer_addr, (const void *)&signal, sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, cstrm);
        NVSHMEMI_NZ_EXIT(status, "cudaMemcpyAsync() failed\n");
    } else {
        call_nvshmemi_signal_op_kernel(sig_addr, signal, sig_op, pe, cstrm);
    }
}

void nvshmemx_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm) {
    nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, cstrm);
}

uint64_t nvshmem_signal_fetch(uint64_t *sig_addr) {
    uint64_t signal;
    CUDA_RUNTIME_CHECK(cudaMemcpy(&signal, sig_addr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return signal;
}
