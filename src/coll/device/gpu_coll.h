/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_COLL_GPU_H_
#define _NVSHMEMI_COLL_GPU_H_

#include "cuda.h"
#include "nvshmem_internal.h"
#include "device/pt-to-pt/proxy_device.cuh"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "device/pt-to-pt/transfer_device.cuh"
#else
#include "device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif

/* structs */

extern int nvshm_gpu_coll_initialized;
extern gpu_coll_env_params_t gpu_coll_env_params_var;

/* function declarations */
extern int nvshmemi_coll_common_gpu_init(void);
extern int nvshmemi_coll_common_gpu_return_modes(void);
extern int nvshmemi_coll_common_gpu_finalize(void);

void nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_t *teami);
void nvshmemi_recexchalgo_free_mem(nvshmemi_team_t *teami);

/* macro definitions */
#define MAX_THREADS_PER_CTA 512

#define NVSHMEMI_COLL_GPU_ERR_POP()                                                \
    do {                                                                           \
        fprintf(stderr, "Error at %s:%d in %s", __FILE__, __LINE__, __FUNCTION__); \
        goto fn_fail;                                                              \
    } while (0)

#endif /* NVSHMEMI_COLL_GPU_H */
