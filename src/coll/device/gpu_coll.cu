/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemx_error.h"
#include "gpu_coll.h"
#include "cpu_coll.h"
#include "nvshmem_internal.h"
#include <map>

using namespace std;

gpu_coll_env_params_t gpu_coll_env_params_var;
int nvshm_gpu_coll_initialized = 0;
map<string, size_t> nvshmemi_alltoall_maxblocksize;
map<string, size_t> nvshmemi_fcollect_maxblocksize;
size_t nvshmemi_barrier_maxblocksize;
map<pair<string, rdxn_ops_t>, size_t> nvshmemi_reduce_maxblocksize;
map<string, size_t> nvshmemi_broadcast_maxblocksize;

int nvshmemi_coll_common_gpu_read_env() {
    int status = 0;

    gpu_coll_env_params_var.reduce_recexch_kval = nvshmemi_options.REDUCE_RECEXCH_KVAL;

    if (gpu_coll_env_params_var.reduce_recexch_kval > nvshmemi_state->npes)
        gpu_coll_env_params_var.reduce_recexch_kval = max(2, nvshmemi_state->npes);

    gpu_coll_env_params_var.bcast_tree_kval = nvshmemi_options.BCAST_TREE_KVAL;
    assert(nvshmemi_options.BCAST_TREE_KVAL >= 2);

    gpu_coll_env_params_var.bcast_algo = nvshmemi_options.BCAST_ALGO;
    gpu_coll_env_params_var.reduce_algo = nvshmemi_options.REDMAXLOC_ALGO;
    return status;
}

int nvshmemi_coll_common_gpu_init_memory() {
    int status = 0;
    nvshmemi_device_state.gpu_coll_env_params_var = gpu_coll_env_params_var;

    /*add barrier so that ipsync arrays aren't touched by other procs*/
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    assert(status == 0);

    return status;
}

int nvshmemi_coll_common_gpu_init() {
    int status = 0;

    // Read env var list to select appropriate designs and

    /* FIXME: Should this become an env var? */
    gpu_coll_env_params_var.gpu_intm_rdxn_size = -1;
    gpu_coll_env_params_var.reduce_recexch_kval = 2;

    status = nvshmemi_coll_common_gpu_read_env();
    if (status) NVSHMEMI_COLL_GPU_ERR_POP();

    status = nvshmemi_coll_common_gpu_init_memory();
    if (status) NVSHMEMI_COLL_GPU_ERR_POP();

    nvshm_gpu_coll_initialized = 1;
    goto fn_out;
fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_gpu_finalize() {
    int status = 0;

    if (0 == nvshm_gpu_coll_initialized) return status;

    nvshm_gpu_coll_initialized = 0;

    goto fn_out;
fn_out:
    return status;
}
