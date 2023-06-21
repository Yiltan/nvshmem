/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "device/pt-to-pt/proxy_device.cuh"

__global__ void nvshmemi_proxy_rma_entrypoint(void *rptr, void *lptr, rma_bytesdesc_t bytesdesc,
                                              int pe, nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_proxy_rma_nbi((void *)rptr, (void *)lptr, bytesdesc.nelems * bytesdesc.elembytes, pe,
                           desc);
#endif
}

__global__ void nvshmemi_proxy_rma_entrypoint_blocking(void *rptr, void *lptr,
                                                       rma_bytesdesc_t bytesdesc, int pe,
                                                       nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_proxy_rma_nbi((void *)rptr, (void *)lptr, bytesdesc.nelems * bytesdesc.elembytes, pe,
                           desc);
    nvshmemi_proxy_quiet(true);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_proxy_rma_nbi((void *)rptr, (void *)lptr, bytesdesc.nelems * bytesdesc.elembytes, pe,
                           desc);
    nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint_blocking(void *rptr, void *lptr,
                                                              rma_bytesdesc_t bytesdesc,
                                                              uint64_t *sig_addr, uint64_t signal,
                                                              int sig_op, int pe,
                                                              nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_proxy_rma_nbi((void *)rptr, (void *)lptr, bytesdesc.nelems * bytesdesc.elembytes, pe,
                           desc);
    nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
    nvshmemi_proxy_quiet(true);
#endif
}

int nvshmemi_proxy_rma_launcher(void *args[], cudaStream_t cstrm, bool is_nbi, bool is_signal) {
    if (is_signal && is_nbi) {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_signal_entrypoint, 1, 1, args, 0,
                                cstrm);
    } else if (is_nbi) {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_entrypoint, 1, 1, args, 0, cstrm);
    } else if (is_signal) {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_signal_entrypoint_blocking, 1, 1,
                                args, 0, cstrm);
    } else {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_entrypoint_blocking, 1, 1, args, 0,
                                cstrm);
    }
}
