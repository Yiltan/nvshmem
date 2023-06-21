/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef NVSHMEM_BOOTSTRAP_DEFINES_H
#define NVSHMEM_BOOTSTRAP_DEFINES_H

typedef struct bootstrap_handle {
    int pg_rank;
    int pg_size;
    int mype_node;
    int npes_node;
    int (*allgather)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int bytes, struct bootstrap_handle *handle);
    int (*barrier)(struct bootstrap_handle *handle);
    void (*global_exit)(int status);
    int (*finalize)(struct bootstrap_handle *handle);
} bootstrap_handle_t;

#endif
