#ifndef NVSHMEM_TYPES_H
#define NVSHMEM_TYPES_H

typedef int32_t nvshmem_team_t;

typedef struct {
    int num_contexts;
} nvshmem_team_config_t;

typedef struct {
    int step1_sendto;
    int* step1_recvfrom;
    int step1_nrecvs;
    int** step2_nbrs;
    int step2_nphases;
} nvshmemi_reduce_recexch_t;

typedef struct {
    int my_pe;
    int start, stride, size;
    int team_idx;
    nvshmem_team_config_t config;
    long config_mask;
    void* nccl_comm; /* To be cast to ncclComm_t whenever used */
    nvshmemi_reduce_recexch_t reduce_recexch;
    size_t rdxn_count;
    uint32_t ll_flag;
    uint64_t alltoall_pwrk[2];
    uint64_t alltoall_count;
    uint64_t bcast_count;
    uint64_t bcast_sync_offset;
    uint64_t fcollect_count;
    uint32_t fcollect_ll_flag;
    bool are_gpus_p2p_connected;
    bool is_team_node;
    nvshmem_team_t team_node;
    bool is_team_same_mype_node;
    nvshmem_team_t team_same_mype_node;
    nvshmemi_pe_dist_t pe_dist;
    /*size_t                       contexts_len;
    struct shmem_transport_ctx_t **contexts;*/
} nvshmemi_team_t;

typedef struct gpu_coll_env_params {
    int gpu_intm_rdxn_size;
    int reduce_recexch_kval;
    int bcast_tree_kval;
    int bcast_algo;
    int reduce_algo;
} gpu_coll_env_params_t;

#endif /* NVSHMEM_TYPES_H */
