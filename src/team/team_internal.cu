/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_coll.h"
#include "nvshmem_internal.h"
#include "nvshmemi_team.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "util.h"
#include "gpu_coll.h"
#include "cpu_coll.h"

long nvshmemi_max_teams;

nvshmemi_team_t nvshmemi_team_world;
nvshmemi_team_t nvshmemi_team_shared;
nvshmemi_team_t nvshmemi_team_node;
nvshmemi_team_t nvshmemi_team_same_mype_node;
nvshmemi_team_t nvshmemi_team_same_gpu;
nvshmemi_team_t nvshmemi_team_gpu_leaders;

nvshmemi_team_t **nvshmemi_team_pool;
long *nvshmemi_psync_pool;
long *nvshmemi_sync_counter;

nvshmemi_team_t **nvshmemi_device_team_pool;

static unsigned char *psync_pool_avail;
static unsigned char *psync_pool_avail_reduced;
static unsigned char *device_psync_pool_avail;
static unsigned char *device_psync_pool_avail_reduced;

static int *team_ret_val;
static int *team_ret_val_reduced;
static int *device_team_ret_val;
static int *device_team_ret_val_reduced;

/* Checks whether a PE has a consistent stride given (start, stride, size).
 * This function is useful within a loop across PE IDs, and sets 'start',
 * 'stride' and 'size' accordingly upon exiting the loop. It also assumes
 * 'start' and 'stride' are initialized to a negative number and 'size' to 0.
 * If an inconsistent stride is found, returns -1. */
static inline int check_for_linear_stride(int pe, int *start, int *stride, int *size) {
    if (*start < 0) {
        *start = pe;
        (*size)++;
    } else if (*stride < 0) {
        *stride = pe - *start;
        (*size)++;
    } else if ((pe - *start) % *stride != 0) {
        NVSHMEMI_WARN_PRINT("Detected non-uniform stride inserting PE %d into <%d, %d, %d>\n", pe,
                            *start, *stride, *size);
        return -1;
    } else {
        (*size)++;
    }
    return 0;
}

#ifndef __CUDA_ARCH__
int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team) {
    int src_pe_world, dest_pe = -1;

    if (src_pe > src_team->size) return -1;

    src_pe_world = src_team->start + src_pe * src_team->stride;
    assert(src_pe_world >= src_team->start && src_pe_world < nvshmemi_state->npes);

    dest_pe = nvshmemi_pe_in_active_set(src_pe_world, dest_team->start, dest_team->stride,
                                        dest_team->size);

    return dest_pe;
}

static inline size_t get_fcollect_psync_len_per_team() {
    size_t fcollect_ll_threshold = nvshmemi_device_state.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_state->npes * fcollect_ll_threshold) / sizeof(long);
    assert(fcollect_ll_threshold % sizeof(long) == 0);

    return fcollect_sync_size;
}

static inline size_t get_psync_len_per_team() {
    size_t fcollect_sync_size = get_fcollect_psync_len_per_team();
    /* sync: Two buffers are used - one for sync/barrier collective ops, the second one during team
       split operation reduce: Two pWrk's are used alternatively across consecutive reduce calls,
       this is to avoid having to put a barrier in between bcast: The buffer is split to do multiple
       consecutive broadcast, when all buffers are used, a barrier is called and then again we begin
       from the start of the buffer fcollect: Two sets of buffer are used to alternate between -
       same way as in reduce. The other fator of 2 is because when using LL double the space is
       needed to fuse flag with data */

    return (2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE +
            NVSHMEMI_BCAST_SYNC_SIZE + fcollect_sync_size + 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE);
}
#endif

size_t nvshmemi_get_teams_mem_requirement() {
    return sizeof(long) * nvshmemi_max_teams * get_psync_len_per_team() + /* psync's */
           2 * N_PSYNC_BYTES +                                            /* psync_pool_avail */
           2 * sizeof(int) +                                              /* team_ret_val */
           2 * sizeof(long) * nvshmemi_max_teams                          /* storing counters */
#ifdef NVSHMEM_USE_NCCL
           + sizeof(ncclUniqueId)
#endif
        ;
}

#ifdef NVSHMEM_USE_NCCL
void nvshmemi_team_init_nccl_comm(nvshmemi_team_t *teami) {
    ncclUniqueId Id;
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    long *pWrk = nvshmemi_team_get_psync(teami, REDUCE);
    if (teami->my_pe == 0) {
        NCCL_CHECK(nccl_ftable.GetUniqueId(&Id));
        CUDA_RUNTIME_CHECK(cudaMemcpy(pWrk, &Id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < size; i++) {
            nvshmem_char_put_nbi((char *)pWrk, (const char *)pWrk, sizeof(ncclUniqueId),
                                 start + i * stride);
        }
        nvshmemi_barrier(teami->team_idx);
    } else {
        nvshmemi_barrier(teami->team_idx);
        CUDA_RUNTIME_CHECK(cudaMemcpy(&Id, pWrk, sizeof(ncclUniqueId), cudaMemcpyDeviceToHost));
    }
    INFO(NVSHMEM_TEAM, "Calling ncclCommInitRank, teami->size = %d, teami->my_pe = %d", teami->size,
         teami->my_pe);
    NCCL_CHECK(
        nccl_ftable.CommInitRank((ncclComm_t *)&teami->nccl_comm, teami->size, Id, teami->my_pe));
}
#endif /* NVSHMEM_USE_NCCL */
void nvshmemi_team_set_p2p_connectivity(nvshmemi_team_t *teami) {
    teami->are_gpus_p2p_connected = 1;
    for (int pe = teami->start; pe < teami->start + teami->stride * teami->size;
         pe += teami->stride) {
        if (nvshmemi_state->peer_heap_base[pe] == NULL) {
            teami->are_gpus_p2p_connected = 0;
            break;
        }
    }
}
/* Team Management Routines */

int nvshmemi_team_init(void) {
    long psync_len;
    int start, stride, size;
    int *scratch = NULL;
    int status = 0;
    nvshmem_transport_pe_info_t *pe_info;

    nvshmemi_max_teams = nvshmemi_options.MAX_TEAMS;
    /* Initialize NVSHMEM_TEAM_WORLD */
    nvshmemi_team_world.team_idx = NVSHMEM_TEAM_WORLD_INDEX;
    nvshmemi_team_world.start = 0;
    nvshmemi_team_world.stride = 1;
    nvshmemi_team_world.size = nvshmemi_state->npes;
    nvshmemi_team_world.my_pe = nvshmemi_state->mype;
    nvshmemi_team_world.rdxn_count = 0;
    nvshmemi_team_world.config_mask = 0;
    memset(&nvshmemi_team_world.config, 0, sizeof(nvshmem_team_config_t));
    nvshmemi_team_world.ll_flag = 1;
    nvshmemi_team_world.bcast_count = 0;
    nvshmemi_team_world.bcast_sync_offset = 0;
    nvshmemi_team_world.fcollect_count = 0;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_world);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_world);
    nvshmemi_team_world.is_team_node = false;
    nvshmemi_team_world.is_team_same_mype_node = false;

    /* Initialize NVSHMEM_TEAM_SHARED */
    nvshmemi_team_shared.team_idx = NVSHMEM_TEAM_SHARED_INDEX;
    nvshmemi_team_shared.my_pe = 0;
    nvshmemi_team_shared.rdxn_count = 0;
    nvshmemi_team_shared.config_mask = 0;
    memset(&nvshmemi_team_shared.config, 0, sizeof(nvshmem_team_config_t));

    nvshmemi_team_shared.start = nvshmemi_state->mype;
    nvshmemi_team_shared.stride = 1;
    nvshmemi_team_shared.size = 1;
    nvshmemi_team_shared.ll_flag = 1;
    nvshmemi_team_shared.bcast_count = 0;
    nvshmemi_team_shared.bcast_sync_offset = 0;
    nvshmemi_team_shared.fcollect_count = 0;
    nvshmemi_team_shared.are_gpus_p2p_connected = 0;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_shared);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_shared);
    INFO(NVSHMEM_INIT, "NVSHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d",
         nvshmemi_team_shared.start, nvshmemi_team_shared.stride, nvshmemi_team_shared.size);
    nvshmemi_team_shared.is_team_node = true;
    nvshmemi_team_shared.is_team_same_mype_node = false;

    /* Initialize NVSHMEM_TEAM_NODE */
    nvshmemi_team_node.team_idx = NVSHMEM_TEAM_NODE_INDEX;
    nvshmemi_team_world.team_node = nvshmemi_team_node.team_idx;
    nvshmemi_team_node.my_pe = nvshmemi_state->mype_node;
    nvshmemi_team_node.rdxn_count = 0;
    nvshmemi_team_node.config_mask = 0;
    memset(&nvshmemi_team_node.config, 0, sizeof(nvshmem_team_config_t));
    nvshmemi_team_node.ll_flag = 1;
    nvshmemi_team_node.bcast_count = 0;
    nvshmemi_team_node.bcast_sync_offset = 0;
    nvshmemi_team_node.fcollect_count = 0;

    uint64_t myHostHash = getHostHash();
    uint64_t *hostHash = (uint64_t *)malloc(sizeof(uint64_t) * nvshmemi_state->npes);
    NVSHMEMI_NULL_ERROR_JMP(hostHash, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "hostHash allocation failed \n");
    status = nvshmemi_boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "allgather of host hashes failed\n");

    /* Search for on-node peer PEs while checking for a consistent stride */
    start = -1;
    stride = -1;
    size = 0;

    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (hostHash[pe] != myHostHash) continue;

        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            start = nvshmemi_state->mype;
            stride = 1;
            size = 1;
            break;
        }
    }
    assert(start >= 0 && size > 0);
    nvshmemi_team_node.start = start;
    nvshmemi_team_node.stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_node.size = size;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_node);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_node);
    nvshmemi_team_node.is_team_node = true;
    nvshmemi_team_node.is_team_same_mype_node = false;

    INFO(NVSHMEM_INIT, "NVSHMEMX_TEAM_NODE: start=%d, stride=%d, size=%d", nvshmemi_team_node.start,
         nvshmemi_team_node.stride, nvshmemi_team_node.size);

    /* Initialize NVSHMEMX_TEAM_SAME_MYPE_NODE */
    nvshmemi_team_same_mype_node.team_idx = NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX;
    nvshmemi_team_world.team_same_mype_node = nvshmemi_team_same_mype_node.team_idx;
    nvshmemi_team_same_mype_node.my_pe = nvshmemi_state->mype / nvshmemi_state->npes_node;
    nvshmemi_team_same_mype_node.rdxn_count = 0;
    nvshmemi_team_same_mype_node.config_mask = 0;
    memset(&nvshmemi_team_same_mype_node.config, 0, sizeof(nvshmem_team_config_t));

    nvshmemi_team_same_mype_node.start = nvshmemi_state->mype_node;
    nvshmemi_team_same_mype_node.stride = nvshmemi_state->npes_node;
    nvshmemi_team_same_mype_node.size = nvshmemi_state->npes / nvshmemi_state->npes_node;
    assert(nvshmemi_state->npes % nvshmemi_state->npes_node == 0);
    nvshmemi_team_same_mype_node.ll_flag = 1;
    nvshmemi_team_same_mype_node.bcast_count = 0;
    nvshmemi_team_same_mype_node.bcast_sync_offset = 0;
    nvshmemi_team_same_mype_node.fcollect_count = 0;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_same_mype_node);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_same_mype_node);
    nvshmemi_team_same_mype_node.is_team_node = false;
    nvshmemi_team_same_mype_node.is_team_same_mype_node = true;
    INFO(NVSHMEM_INIT, "NVSHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d",
         nvshmemi_team_same_mype_node.start, nvshmemi_team_same_mype_node.stride,
         nvshmemi_team_same_mype_node.size);

    /* Initialize team NVSHMEMI_TEAM_SAME_GPU */
    nvshmemi_team_same_gpu.team_idx = NVSHMEM_TEAM_SAME_GPU_INDEX;
    nvshmemi_team_same_gpu.rdxn_count = 0;
    nvshmemi_team_same_gpu.ll_flag = 1;
    nvshmemi_team_same_gpu.bcast_count = 0;
    nvshmemi_team_same_gpu.bcast_sync_offset = 0;
    nvshmemi_team_same_gpu.fcollect_count = 0;
    nvshmemi_team_same_gpu.config_mask = 0;
    memset(&nvshmemi_team_same_gpu.config, 0, sizeof(nvshmem_team_config_t));
    pe_info = nvshmemi_state->pe_info;
    start = -1;
    stride = -1;
    size = 0;
    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (pe_info[pe].hostHash != pe_info[nvshmemi_state->mype].hostHash ||
            memcmp(&pe_info[pe].gpu_uuid, &pe_info[nvshmemi_state->mype].gpu_uuid,
                   sizeof(cudaUUID_t)) != 0)
            continue;

        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            NVSHMEMI_ERROR_EXIT("Could not form NVSHMEMI_TEAM_SAME_GPU\n");
            break;
        }
    }
    assert(start >= 0 && size > 0);
    nvshmemi_team_same_gpu.my_pe = (nvshmemi_state->mype - start) / stride;
    nvshmemi_team_same_gpu.start = start;
    nvshmemi_team_same_gpu.stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_same_gpu.size = size;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_same_gpu);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_same_gpu);
    nvshmemi_team_same_gpu.is_team_node = true;
    nvshmemi_team_same_gpu.is_team_same_mype_node = false;
    INFO(NVSHMEM_INIT, "NVSHMEMI_TEAM_SAME_GPU: start=%d, stride=%d, size=%d",
         nvshmemi_team_same_gpu.start, nvshmemi_team_same_gpu.stride, nvshmemi_team_same_gpu.size);

    /* All GPUs must have same number of processes (requires for us to form teams) */

    /* Initialize team NVSHMEMI_TEAM_GPU_LEADERS */
    scratch = (int *)malloc(sizeof(int) * nvshmemi_state->npes);
    NVSHMEMI_NULL_ERROR_JMP(scratch, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "Unable to allocate host memory for team creation.\n");
    if (nvshmemi_team_same_gpu.start ==
        nvshmemi_state->mype) { /* Only GPU leaders are part of this team */
        nvshmemi_team_gpu_leaders.team_idx = NVSHMEM_TEAM_GPU_LEADERS_INDEX;
        nvshmemi_team_gpu_leaders.config_mask = 0;
        memset(&nvshmemi_team_gpu_leaders.config, 0, sizeof(nvshmem_team_config_t));

        nvshmemi_team_gpu_leaders.start = 0;
        nvshmemi_team_gpu_leaders.stride =
            (nvshmemi_team_same_gpu.stride == 1) ? nvshmemi_team_same_gpu.size : 1;
        nvshmemi_team_gpu_leaders.size = nvshmemi_state->npes / nvshmemi_team_same_gpu.size;
        nvshmemi_team_gpu_leaders.my_pe = (nvshmemi_state->mype - nvshmemi_team_gpu_leaders.start) /
                                          nvshmemi_team_gpu_leaders.stride;
        nvshmemi_team_gpu_leaders.rdxn_count = 0;
        nvshmemi_team_gpu_leaders.ll_flag = 1;
        nvshmemi_team_gpu_leaders.bcast_count = 0;
        nvshmemi_team_gpu_leaders.bcast_sync_offset = 0;
        nvshmemi_team_gpu_leaders.fcollect_count = 0;
        nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_gpu_leaders);
        nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_gpu_leaders);
        status =
            nvshmemi_boot_handle.allgather((void *)&nvshmemi_team_gpu_leaders.my_pe,
                                           (void *)scratch, sizeof(int), &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "allgather of gpu leaders failed\n");
        /* Check whether a valid TEAM_GPU_LEADERS was formed */
        int last_mype = -1;
        for (int i = 0; i < nvshmemi_state->npes; i++) {
            if (scratch[i] != -1) {
                if (scratch[i] != last_mype + 1) {
                    WARN(
                        "NVSHMEMI_TEAM_GPU_LEADERS could not be formed, Limited MPG support will "
                        "not be available\n");
                    break;
                } else {
                    last_mype++;
                }
            }
        }
        /* XXX: Note that we are not setting team_node and team_same_mype_node for
         * nvshmemi_team_gpu_leaders */
        nvshmemi_team_gpu_leaders.is_team_node = false;
        nvshmemi_team_gpu_leaders.is_team_same_mype_node = false;
        INFO(NVSHMEM_INIT, "NVSHMEMI_TEAM_GPU_LEADERS: start=%d, stride=%d, size=%d",
             nvshmemi_team_gpu_leaders.start, nvshmemi_team_gpu_leaders.stride,
             nvshmemi_team_gpu_leaders.size);
    } else {
        int my_pe = -1;
        status = nvshmemi_boot_handle.allgather((void *)&my_pe, (void *)scratch, sizeof(int),
                                                &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "allgather of gpu leaders failed\n");
    }
    if (nvshmemi_max_teams < NVSHMEM_TEAMS_MIN) nvshmemi_max_teams = NVSHMEM_TEAMS_MIN;

    if (nvshmemi_max_teams > N_PSYNC_BYTES * CHAR_BIT) {
        NVSHMEMI_ERROR_EXIT("Requested %ld teams, but only %d are supported\n", nvshmemi_max_teams,
                            N_PSYNC_BYTES * CHAR_BIT);
        goto cleanup;
    }

    nvshmemi_team_pool = (nvshmemi_team_t **)malloc(nvshmemi_max_teams * sizeof(nvshmemi_team_t *));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_team_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_team_pool allocation failed \n");
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_pool,
                                  nvshmemi_max_teams * sizeof(nvshmemi_team_t *)));
    nvshmemi_device_state.team_pool = nvshmemi_device_team_pool;

    for (long i = 0; i < nvshmemi_max_teams; i++) {
        nvshmemi_team_pool[i] = NULL;
    }

    nvshmemi_init_array_kernel<nvshmemi_team_t *>
        <<<1, 1>>>(nvshmemi_device_team_pool, nvshmemi_max_teams, NULL);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    nvshmemi_team_pool[NVSHMEM_TEAM_WORLD_INDEX] = &nvshmemi_team_world;
    nvshmemi_team_pool[NVSHMEM_TEAM_SHARED_INDEX] = &nvshmemi_team_shared;
    nvshmemi_team_pool[NVSHMEM_TEAM_NODE_INDEX] = &nvshmemi_team_node;
    nvshmemi_team_pool[NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX] = &nvshmemi_team_same_mype_node;
    nvshmemi_team_pool[NVSHMEM_TEAM_SAME_GPU_INDEX] = &nvshmemi_team_same_gpu;
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype)
        nvshmemi_team_pool[NVSHMEM_TEAM_GPU_LEADERS_INDEX] = &nvshmemi_team_gpu_leaders;

    /* Allocate pSync pool, each with the maximum possible size requirement */
    /* Create two pSyncs per team for back-to-back collectives and one for barriers.
     * Array organization:
     *
     * [ (world) (shared) (team 1) (team 2) ...  (world) (shared) (team 1) (team 2) ... ]
     *  <----------- groups 1 & 2-------------->|<------------- group 3 ---------------->
     *  <--- (bcast, collect, reduce, etc.) --->|<------ (barriers and syncs) ---------->
     * */
    psync_len = nvshmemi_max_teams * get_psync_len_per_team();
    nvshmemi_psync_pool = (long *)nvshmemi_malloc(sizeof(long) * psync_len);
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_psync_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_psync_pool allocation failed \n");

    nvshmemi_device_state.psync_pool = nvshmemi_psync_pool;

    nvshmemi_init_array_kernel<long><<<1, 1>>>(nvshmemi_psync_pool, psync_len, NVSHMEMI_SYNC_VALUE);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    nvshmemi_sync_counter = (long *)nvshmemi_malloc(2 * nvshmemi_max_teams * sizeof(long));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_sync_counter, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_sync_counter allocation failed \n");

    nvshmemi_device_state.sync_counter = nvshmemi_sync_counter;
    nvshmemi_set_device_state(&nvshmemi_device_state);

    nvshmemi_init_array_kernel<long><<<1, 1>>>(nvshmemi_sync_counter, 2 * nvshmemi_max_teams, 1);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    /* Convenience pointer to the group-3 pSync array (for barriers and syncs): */
    psync_pool_avail = (unsigned char *)malloc(2 * N_PSYNC_BYTES);
    NVSHMEMI_NULL_ERROR_JMP(psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "psync_pool_avail allocation failed \n");
    psync_pool_avail_reduced = &psync_pool_avail[N_PSYNC_BYTES];

    device_psync_pool_avail = (unsigned char *)nvshmemi_malloc(2 * N_PSYNC_BYTES);
    NVSHMEMI_NULL_ERROR_JMP(device_psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "device_psync_pool_avail allocation failed \n");
    device_psync_pool_avail_reduced = &device_psync_pool_avail[N_PSYNC_BYTES];
    /* Initialize the psync bits to 1, making all slots available: */
    memset(psync_pool_avail, 0, 2 * N_PSYNC_BYTES);
    for (size_t i = 0; i < (size_t)nvshmemi_max_teams; i++) {
        nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, i);
    }

    /* Set the bits for NVSHMEM_TEAM_WORLD, NVSHMEM_TEAM_SHARED, NVSHMEMX_TEAM_NODE to 0: */
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_WORLD_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SHARED_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_NODE_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SAME_GPU_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_GPU_LEADERS_INDEX);

    /* Initialize an integer used to agree on an equal return value across PEs in team creation: */
    team_ret_val = (int *)malloc(sizeof(int) * 2);
    NVSHMEMI_NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "team_ret_val allocation failed \n");
    team_ret_val_reduced = &team_ret_val[1];

    device_team_ret_val = (int *)nvshmemi_malloc(sizeof(int) * 2);
    NVSHMEMI_NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "device_team_ret_val allocation failed \n");
    device_team_ret_val_reduced = &device_team_ret_val[1];

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

    nvshmemi_team_t *nvshmemi_device_team_world, *nvshmemi_device_team_shared,
        *nvshmemi_device_team_node, *nvshmemi_device_team_same_mype_node,
        *nvshmemi_device_team_same_gpu, *nvshmemi_device_team_gpu_leaders;
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_world, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_world, &nvshmemi_team_world,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_WORLD_INDEX],
                                  &nvshmemi_device_team_world, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_shared, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_shared, &nvshmemi_team_shared,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SHARED_INDEX],
                                  &nvshmemi_device_team_shared, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_node, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_node, &nvshmemi_team_node,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_NODE_INDEX],
                                  &nvshmemi_device_team_node, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&nvshmemi_device_team_same_mype_node, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_same_mype_node,
                                  &nvshmemi_team_same_mype_node, sizeof(nvshmemi_team_t),
                                  cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX],
                                  &nvshmemi_device_team_same_mype_node, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&nvshmemi_device_team_same_gpu, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_same_gpu, &nvshmemi_team_same_gpu,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SAME_GPU_INDEX],
                                  &nvshmemi_device_team_same_gpu, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&nvshmemi_device_team_gpu_leaders, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_gpu_leaders, &nvshmemi_team_gpu_leaders,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_GPU_LEADERS_INDEX],
                                  &nvshmemi_device_team_gpu_leaders, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) {
        /* Setup NCCL usage */
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_world);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_shared);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_node);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_same_mype_node);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_same_gpu);
        if (nvshmemi_pe_in_active_set(nvshmemi_state->mype, nvshmemi_team_gpu_leaders.start,
                                      nvshmemi_team_gpu_leaders.stride,
                                      nvshmemi_team_gpu_leaders.size) >= 0) {
            nvshmemi_team_init_nccl_comm(&nvshmemi_team_gpu_leaders);
        }
    }
#endif /* NVSHMEM_USE_NCCL */

#if defined(NVSHMEM_PPC64LE)
    if (nvshmemi_use_nccl) {
        /* Set GPU thread stack size to be max stack size of any kernel invoked by NCCL.
           The value 1256 has been obtained by profiling all NCCL kernels in NCCL 2.8.3-1.
           This value is being set to prevent any memory config during application run
           as that can lead to potential deadlock */
        if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
            CUDA_RUNTIME_CHECK(
                cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
            if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE < 1256)
                NVSHMEMI_WARN_PRINT(
                    "CUDA stack size limit has been set to less than 1256.\n"
                    "This can lead to hangs because a NCCL kernel can need up\n"
                    "to 1256 bytes");
        } else
            CUDA_RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 1256));
    } else if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
        CUDA_RUNTIME_CHECK(
            cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
    }
#endif

cleanup:
    if (scratch) {
        free(scratch);
    }
    if (hostHash) {
        free(hostHash);
    }

    if (status != NVSHMEMX_SUCCESS) {
        if (nvshmemi_team_pool) {
            free(nvshmemi_team_pool);
            nvshmemi_team_pool = NULL;
            cudaFree(nvshmemi_device_team_pool);
            nvshmemi_device_team_pool = NULL;
        }
        if (nvshmemi_psync_pool) {
            nvshmemi_free(nvshmemi_psync_pool);
            nvshmemi_psync_pool = NULL;
        }
        if (psync_pool_avail) {
            free(psync_pool_avail);
            psync_pool_avail = NULL;
        }
        if (device_psync_pool_avail) {
            nvshmemi_free(device_psync_pool_avail);
            device_psync_pool_avail = NULL;
        }
        if (team_ret_val) {
            free(team_ret_val);
            team_ret_val = NULL;
        }
        if (device_team_ret_val) {
            nvshmemi_free(device_team_ret_val);
            device_team_ret_val = NULL;
        }
    }

    return status;
}

int nvshmemi_team_finalize(void) {
    /* Destroy all undestroyed teams */
    for (long i = 0; i < nvshmemi_max_teams; i++) {
        if (nvshmemi_team_pool[i] != NULL) nvshmemi_team_destroy(nvshmemi_team_pool[i]);
    }

    free(nvshmemi_team_pool);
    CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_team_pool));

    nvshmemi_free(nvshmemi_psync_pool);
    nvshmemi_free(nvshmemi_sync_counter);

    free(psync_pool_avail);
    nvshmemi_free(device_psync_pool_avail);
    free(team_ret_val);
    nvshmemi_free(device_team_ret_val);

    return 0;
}

int nvshmemi_team_split_strided(nvshmemi_team_t *parent_team, int PE_start, int PE_stride,
                                int PE_size, const nvshmem_team_config_t *config, long config_mask,
                                nvshmem_team_t *new_team) {
    *new_team = NVSHMEM_TEAM_INVALID;
    nvshmem_barrier(parent_team->team_idx);

    int global_PE_start = nvshmemi_team_pe(parent_team, PE_start);
    int global_PE_end = global_PE_start + PE_stride * (PE_size - 1);

    if (PE_start < 0 || PE_start >= parent_team->size || PE_size <= 0 ||
        PE_size > parent_team->size || PE_stride < 1) {
        NVSHMEMI_WARN_PRINT(
            "Invalid <start, stride, size>: child <%d, %d, %d>, parent <%d, %d, %d>\n", PE_start,
            PE_stride, PE_size, parent_team->start, parent_team->stride, parent_team->size);
        return -1;
    }

    if (global_PE_start >= nvshmemi_state->npes || global_PE_end >= nvshmemi_state->npes) {
        NVSHMEMI_WARN_PRINT("Starting PE (%d) or ending PE (%d) is invalid\n", global_PE_start,
                            global_PE_end);
        return -1;
    }

    int my_pe =
        nvshmemi_pe_in_active_set(nvshmemi_state->mype, global_PE_start, PE_stride, PE_size);

    long *psync_reduce = nvshmemi_team_get_psync(parent_team, REDUCE);
    long *psync = &nvshmemi_team_get_psync(parent_team, SYNC)[NVSHMEMI_SYNC_SIZE];
    long *sync_counter = &nvshmemi_team_get_sync_counter(parent_team)[1];
    nvshmemi_team_t *myteam = NULL;
    *team_ret_val = 0;
    *team_ret_val_reduced = 0;

    if (my_pe >= 0) {
        char bit_str[NVSHMEMI_DIAG_STRLEN];

        myteam = (nvshmemi_team_t *)calloc(1, sizeof(nvshmemi_team_t));

        myteam->my_pe = my_pe;
        myteam->start = global_PE_start;
        myteam->stride = PE_stride;
        myteam->size = PE_size;
        myteam->rdxn_count = 0;
        myteam->ll_flag = 1;
        myteam->bcast_count = 0;
        myteam->bcast_sync_offset = 0;
        myteam->fcollect_count = 0;
        if (config) {
            myteam->config = *config;
            myteam->config_mask = config_mask;
        }
        myteam->team_idx = -1;
        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail, N_PSYNC_BYTES);

        CUDA_RUNTIME_CHECK(cudaMemcpy(device_psync_pool_avail, psync_pool_avail, N_PSYNC_BYTES,
                                      cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        nvshmemi_reduce_kernel<unsigned char, RDXN_OPS_AND>
            <<<1, 1>>>(myteam->start, myteam->stride, myteam->size,
                       (unsigned char *)device_psync_pool_avail_reduced,
                       (const unsigned char *)device_psync_pool_avail, N_PSYNC_BYTES,
                       (unsigned char *)psync_reduce, (long *)(psync), sync_counter);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        CUDA_RUNTIME_CHECK(cudaMemcpy(psync_pool_avail_reduced, device_psync_pool_avail_reduced,
                                      N_PSYNC_BYTES, cudaMemcpyDeviceToHost));

        /* We cannot release the psync here, because this reduction may not
         * have been performed on the entire parent team. */
        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                               N_PSYNC_BYTES);

        /* Select the least signficant nonzero bit, which corresponds to an available pSync. */
        myteam->team_idx = nvshmemi_bit_1st_nonzero(psync_pool_avail_reduced, N_PSYNC_BYTES);

        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                               N_PSYNC_BYTES);
        if (myteam->team_idx == -1 || myteam->team_idx >= (int)nvshmemi_max_teams) {
            NVSHMEMI_WARN_PRINT(
                "No more teams available (max = %ld), try setting NVSHMEM_MAX_TEAMS environment "
                "variable\n",
                nvshmemi_max_teams);
            /* No psync was available, but must call barrier across parent team before returning. */
            myteam->team_idx = -1;
            *team_ret_val = 1;
        } else {
            /* Set the selected psync bit to 0, reserving that slot */
            nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, myteam->team_idx);

            *new_team = myteam->team_idx;

            nvshmemi_team_pool[myteam->team_idx] = myteam;
            nvshmemi_team_t *device_team_addr;
            CUDA_RUNTIME_CHECK(cudaMalloc((void **)&device_team_addr, sizeof(nvshmemi_team_t)));
            nvshmemi_team_set_p2p_connectivity(myteam);
            nvshmemi_recexchalgo_get_neighbors(myteam);
            CUDA_RUNTIME_CHECK(cudaMemcpy(device_team_addr, myteam, sizeof(nvshmemi_team_t),
                                          cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[myteam->team_idx],
                                          &device_team_addr, sizeof(nvshmemi_team_t *),
                                          cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
#ifdef NVSHMEM_USE_NCCL
            if (nvshmemi_use_nccl) nvshmemi_team_init_nccl_comm(myteam);
#endif

            /* Build team_node */
            myteam->is_team_node = false;
            int i;
            for (i = 1; i < myteam->size; i++) {
                if (nvshmemi_host_hashes[myteam->start] !=
                    nvshmemi_host_hashes[myteam->start + i * myteam->stride]) {
                    break;
                }
            }
            if (i == myteam->size) myteam->is_team_node = true;

            myteam->is_team_same_mype_node = true;
            for (int i = 0; i < myteam->size; i++) {
                for (int j = i + 1; j < myteam->size; j++) {
                    if (nvshmemi_host_hashes[myteam->start + i * myteam->stride] ==
                        nvshmemi_host_hashes[myteam->start + j * myteam->stride]) {
                        myteam->is_team_same_mype_node = false;
                    }
                }
            }

            /* count PEs on the same node */
            int team_npes_node = 0;
            for (int i = 0; i < myteam->size; i++) {
                if (nvshmemi_team_translate_pe(myteam, i, &nvshmemi_team_node) != -1) {
                    team_npes_node++;
                }
            }
            if (!myteam->is_team_node && !myteam->is_team_same_mype_node) {
                /* Now I am just going to repurpose device_psync_pool_avail symm memory for the
                   purpose of finding max of team_npes_node */
                assert(sizeof(int) <= N_PSYNC_BYTES);
                CUDA_RUNTIME_CHECK(cudaMemcpy(device_psync_pool_avail, &team_npes_node, sizeof(int),
                                              cudaMemcpyHostToDevice));
                CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
                nvshmemi_reduce_kernel<int, RDXN_OPS_MAX><<<1, 1>>>(
                    myteam->start, myteam->stride, myteam->size,
                    (int *)device_psync_pool_avail_reduced, (const int *)device_psync_pool_avail, 1,
                    (int *)psync_reduce, (long *)(psync), sync_counter);
                CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

                CUDA_RUNTIME_CHECK(cudaMemcpy(&team_npes_node, device_psync_pool_avail_reduced,
                                              sizeof(int), cudaMemcpyDeviceToHost));
                nvshmemi_team_split_2d(myteam, team_npes_node, NULL, 0, &myteam->team_node, NULL, 0,
                                       &myteam->team_same_mype_node);
            }
            CUDA_RUNTIME_CHECK(cudaMemcpy(device_team_addr, myteam, sizeof(nvshmemi_team_t),
                                          cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        }
        nvshmemi_init_array_kernel<long><<<1, 1>>>(sync_counter, 1, 1);
        nvshmemi_init_array_kernel<long><<<1, 1>>>(psync, NVSHMEMI_SYNC_SIZE, NVSHMEMI_SYNC_VALUE);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    }

    /* This barrier on the parent team eliminates problematic race conditions
     * during psync allocation between back-to-back team creations. */
    nvshmem_quiet();
    // nvshmem_barrier(parent_team->start, parent_team->stride, parent_team->size, psync);
    nvshmem_team_sync(parent_team->team_idx);
    /* This OR reduction assures all PEs return the same value.  */
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(device_team_ret_val, team_ret_val, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    nvshmemi_call_rdxn_on_stream_kernel<int, RDXN_OPS_MAX>(
        parent_team->team_idx, device_team_ret_val_reduced, device_team_ret_val, 1,
        nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
    CUDA_RUNTIME_CHECK(cudaMemcpy(team_ret_val_reduced, device_team_ret_val_reduced, sizeof(int),
                                  cudaMemcpyDeviceToHost));

    /* If no team was available, print some team triplet info and return nonzero. */
    if (myteam != NULL && myteam->team_idx == -1) {
        NVSHMEMI_WARN_PRINT("Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                            global_PE_start, PE_stride, PE_size, parent_team->start,
                            parent_team->stride, parent_team->size);
        /* TODO: In the event one of the PEs fails to create the team, do we need to revert the team
         * on all of the other ones? */
        free(myteam);
    }

    return *team_ret_val_reduced;
}

int nvshmemi_team_split_2d(nvshmemi_team_t *parent_team, int xrange,
                           const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                           nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                           long yaxis_mask, nvshmem_team_t *yaxis_team) {
    *xaxis_team = NVSHMEM_TEAM_INVALID;
    *yaxis_team = NVSHMEM_TEAM_INVALID;

    if (xrange > parent_team->size) {
        xrange = parent_team->size;
    }

    const int parent_stride = parent_team->stride;
    const int parent_size = parent_team->size;
    const int num_xteams = ceil(parent_size / (float)xrange);
    const int num_yteams = xrange;

    int start = 0;
    int ret = 0;

    for (int i = 0; i < num_xteams; i++) {
        nvshmem_team_t my_xteam;
        int xsize = (i == num_xteams - 1 && parent_size % xrange) ? parent_size % xrange : xrange;
        ret = nvshmemi_team_split_strided(parent_team, start, parent_stride, xsize, xaxis_config,
                                          xaxis_mask, &my_xteam);
        if (ret) {
            NVSHMEMI_ERROR_PRINT("Creation of x-axis team %d of %d failed\n", i + 1, num_xteams);
        }
        start += xrange;

        if (my_xteam != NVSHMEM_TEAM_INVALID) {
            assert(*xaxis_team == NVSHMEM_TEAM_INVALID);
            *xaxis_team = my_xteam;
        }
    }

    start = 0;

    for (int i = 0; i < num_yteams; i++) {
        nvshmem_team_t my_yteam;
        int remainder = parent_size % xrange;
        int yrange = parent_size / xrange;
        int ysize = (remainder && i < remainder) ? yrange + 1 : yrange;

        ret = nvshmemi_team_split_strided(parent_team, start, xrange * parent_stride, ysize,
                                          yaxis_config, yaxis_mask, &my_yteam);
        if (ret) {
            NVSHMEMI_ERROR_PRINT("Creation of y-axis team %d of %d failed\n", i + 1, num_yteams);
        }
        start += 1;

        if (my_yteam != NVSHMEM_TEAM_INVALID) {
            assert(*yaxis_team == NVSHMEM_TEAM_INVALID);
            *yaxis_team = my_yteam;
        }
    }

    nvshmem_quiet();
    nvshmem_team_sync(parent_team->team_idx);

    return 0;
}

void nvshmemi_team_destroy(nvshmemi_team_t *team) {
    int idx = team->team_idx;
    if (nvshmemi_bit_fetch(psync_pool_avail, idx)) {
        NVSHMEMI_ERROR_PRINT("Destroying a team without an active pSync\n");
    }

    /* Since it is a collective routine, perform a barrier */
    // nvshmem_barrier(idx);

    nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, idx);

    nvshmemi_team_pool[idx] = NULL;
    CUDA_RUNTIME_CHECK(cudaMemset(&nvshmemi_device_team_pool[idx], 0, sizeof(nvshmemi_team_t *)));

    nvshmemi_init_array_kernel<long><<<1, 1>>>(&nvshmemi_sync_counter[2 * idx], 2, 1);
    nvshmemi_init_array_kernel<long><<<1, 1>>>(&nvshmemi_psync_pool[idx * get_psync_len_per_team()],
                                               get_psync_len_per_team(), NVSHMEMI_SYNC_VALUE);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    if (team != &nvshmemi_team_world && team != &nvshmemi_team_shared &&
        team != &nvshmemi_team_node && team != &nvshmemi_team_same_mype_node &&
        team != &nvshmemi_team_same_gpu && team != &nvshmemi_team_gpu_leaders) {
        nvshmemi_recexchalgo_free_mem(team);
#ifdef NVSHMEM_USE_NCCL
        if (nvshmemi_use_nccl) NCCL_CHECK(nccl_ftable.CommDestroy((ncclComm_t)team->nccl_comm));
#endif
        free(team);
        nvshmemi_team_t *device_team_addr;
        CUDA_RUNTIME_CHECK(cudaMemcpy((void **)&device_team_addr, &nvshmemi_device_team_pool[idx],
                                      sizeof(nvshmemi_team_t *), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaFree(device_team_addr));
    }
}

#ifndef __CUDA_ARCH__
long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    long *team_psync;
    size_t psync_fcollect_len;
    psync_fcollect_len = get_fcollect_psync_len_per_team();
    team_psync = &nvshmemi_psync_pool[team->team_idx * get_psync_len_per_team()];
    switch (op) {
        case SYNC:
            return team_psync;
        case REDUCE:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               (NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * (team->rdxn_count % 2))];
        case BCAST:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE];
        case FCOLLECT:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE +
                               NVSHMEMI_BCAST_SYNC_SIZE];
        case ALLTOALL:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               (NVSHMEMI_ALLTOALL_SYNC_SIZE * (team->alltoall_count % 2))];
        default:
            printf("Incorrect argument to nvshmemi_team_get_psync\n");
            return NULL;
    }
}

long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team) {
    return &nvshmemi_sync_counter[2 * team->team_idx];
}
#endif
