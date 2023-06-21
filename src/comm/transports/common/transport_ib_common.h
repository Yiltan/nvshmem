/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_IB_COMMON_H
#define _TRANSPORT_IB_COMMON_H

#include "transport_common.h"
#include "infiniband/verbs.h"

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

struct nvshmemt_ib_common_mem_handle {
    int fd;
    uint32_t lkey;
    uint32_t rkey;
    struct ibv_mr *mr;
    void *buf;
    bool local_only;
};

int nvshmemt_ib_common_nv_peer_mem_available();

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable, struct ibv_pd *pd,
                                      nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                      bool local_only, bool dmabuf_support,
                                      struct nvshmemi_cuda_fn_table *table, int log_level);

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level);

#endif
