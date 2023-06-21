/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _P2P_H
#define _P2P_H

#include "transport.h"

#define NVSHMEM_PCIE_DBF_BUFFER_LEN 50

typedef struct {
    int ndev;
    CUdevice *cudev;
    int *devid;
    CUdeviceptr *curetval;
    CUdevice cudevice;
    int device_id;
    uint64_t hostHash;
    pcie_id_t *pcie_ids;
    char pcie_bdf[NVSHMEM_PCIE_DBF_BUFFER_LEN];
} transport_p2p_state_t;

#endif
