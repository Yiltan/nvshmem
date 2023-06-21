/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdint.h>

#ifndef __TRANSPORT_DEFINES_H
#define __TRANSPORT_DEFINES_H

#define NVSHMEM_MEM_HANDLE_SIZE 512

typedef enum {
    NVSHMEMI_OP_PUT = 1,
    NVSHMEMI_OP_P,
    NVSHMEMI_OP_PUT_SIGNAL,
    NVSHMEMI_OP_GET,
    NVSHMEMI_OP_G,
    NVSHMEMI_OP_FENCE,
    NVSHMEMI_OP_AMO,
    NVSHMEMI_OP_QUIET,
} nvshmemi_op_t;

typedef enum {
    NVSHMEMI_AMO_ACK = 1,
    NVSHMEMI_AMO_INC,
    NVSHMEMI_AMO_SET,
    NVSHMEMI_AMO_ADD,
    NVSHMEMI_AMO_AND,
    NVSHMEMI_AMO_OR,
    NVSHMEMI_AMO_XOR,
    NVSHMEMI_AMO_SIGNAL,
    NVSHMEMI_AMO_SIGNAL_SET,
    NVSHMEMI_AMO_SIGNAL_ADD,
    NVSHMEMI_AMO_END_OF_NONFETCH,  // end of nonfetch atomics
    NVSHMEMI_AMO_FETCH,
    NVSHMEMI_AMO_FETCH_INC,
    NVSHMEMI_AMO_FETCH_ADD,
    NVSHMEMI_AMO_FETCH_AND,
    NVSHMEMI_AMO_FETCH_OR,
    NVSHMEMI_AMO_FETCH_XOR,
    NVSHMEMI_AMO_SWAP,
    NVSHMEMI_AMO_COMPARE_SWAP,
} nvshmemi_amo_t;

typedef struct {
    volatile uint64_t data;
    volatile uint64_t flag;
} g_elem_t;

typedef struct pcie_identifier {
    int dev_id;
    int bus_id;
    int domain_id;
} pcie_id_t;

typedef struct nvshmem_mem_handle {
    char reserved[NVSHMEM_MEM_HANDLE_SIZE];
} nvshmem_mem_handle_t;

#endif
