/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_ib_common.h"
#include <unistd.h>
#include <assert.h>

int nvshmemt_ib_common_nv_peer_mem_available() {
    if (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }
    if (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable, struct ibv_pd *pd,
                                      nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                      bool local_only, bool dmabuf_support,
                                      struct nvshmemi_cuda_fn_table *table, int log_level) {
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    struct ibv_mr *mr = NULL;
    int status = 0;

    assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);
#if CUDA_VERSION >= 11070
    bool host_memory = false;
    if (local_only) {
        cudaPointerAttributes attr;
        status = cudaPointerGetAttributes(&attr, buf);
        if (status != cudaSuccess) {
            host_memory = true;
            status = 0;
            cudaGetLastError();
        } else if (attr.type != cudaMemoryTypeDevice) {
            host_memory = true;
        }
    }

    if (ftable->reg_dmabuf_mr != NULL && !host_memory && dmabuf_support &&
        CUPFN(table, cuMemGetHandleForAddressRange)) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        CUdeviceptr p;
        size_t size_aligned = ROUNDUP(length, page_size);
        p = (CUdeviceptr)((uintptr_t)buf & ~(page_size - 1));

        CUCHECKGOTO(table,
                    cuMemGetHandleForAddressRange(&handle->fd, (CUdeviceptr)p, size_aligned,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                    status, out);

        mr = ftable->reg_dmabuf_mr(pd, 0, size_aligned, (uint64_t)p, handle->fd,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if (mr == NULL) {
            goto reg_dmabuf_failure;
        }

        INFO(log_level, "ibv_reg_dmabuf_mr handle %p handle->mr %p", handle, handle->mr);
    } else {
    reg_dmabuf_failure:
#endif
        handle->fd = 0;
        mr = ftable->reg_mr(pd, buf, length,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        NVSHMEMI_NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed \n");
        INFO(log_level, "ibv_reg_mr handle %p handle->mr %p", handle, handle->mr);
#if CUDA_VERSION >= 11070
    }
#endif
    handle->buf = buf;
    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = mr;
    handle->local_only = local_only;

out:
    return status;
}

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level) {
    int status = 0;
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    INFO(log_level, "ibv_dereg_mr handle %p handle->mr %p", handle, handle->mr);
    if (handle->mr) {
        status = ftable->dereg_mr((struct ibv_mr *)handle->mr);
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}
