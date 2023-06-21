/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __TOPO_H
#define __TOPO_H

int nvshmemi_get_device_by_distance(int *device, struct nvshmem_transport *tcurr);
int nvshmemi_detect_same_device(nvshmemi_state_t *state);
int nvshmemi_build_transport_map(nvshmemi_state_t *state);

#endif
