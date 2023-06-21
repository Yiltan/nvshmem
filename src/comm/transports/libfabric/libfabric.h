/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_atomic.h>

#define NVSHMEMT_LIBFABRIC_MAJ_VER 1
#define NVSHMEMT_LIBFABRIC_MIN_VER 5

#define NVSHMEMT_LIBFABRIC_DOMAIN_LEN 32
#define NVSHMEMT_LIBFABRIC_PROVIDER_LEN 32
#define NVSHMEMT_LIBFABRIC_EP_LEN 128

/* one EP for all proxy ops, one for host ops */
#define NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS 2
#define NVSHMEMT_LIBFABRIC_PROXY_EP_IDX 1
#define NVSHMEMT_LIBFABRIC_HOST_EP_IDX 0

#define NVSHMEMT_LIBFABRIC_QUIET_TIMEOUT_MS 20

/* Maximum size of inject data. Currently
 * the max size we will use is one element
 * of a given type. Making it 16 bytes in the
 * case of complex number support. */
#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMT_LIBFABRIC_INJECT_BYTES 16
#else
#define NVSHMEMT_LIBFABRIC_INJECT_BYTES 8
#endif

#define NVSHMEMT_LIBFABRIC_MAX_RETRIES (1ULL << 20)

typedef struct {
    char name[NVSHMEMT_LIBFABRIC_DOMAIN_LEN];
} nvshmemt_libfabric_domain_name_t;

typedef struct {
    char name[NVSHMEMT_LIBFABRIC_EP_LEN];
} nvshmemt_libfabric_ep_name_t;

typedef struct {
    struct fid_ep *endpoint;
    struct fid_cq *cq;
    struct fid_cntr *counter;
    uint64_t submitted_ops;
} nvshmemt_libfabric_endpoint_t;

typedef struct {
    struct fi_info *prov_info;
    struct fi_info *all_prov_info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *addresses;
    nvshmemt_libfabric_endpoint_t *eps;
    /* local_mr is used only for consistency ops. */
    struct fid_mr *local_mr[2];
    uint64_t local_mr_key[2];
    void *local_mr_desc[2];
    void *local_mem_ptr;
    nvshmemt_libfabric_domain_name_t *domain_names;
    int num_domains;
    int next_key;
    int is_verbs;
    int log_level;
    struct nvshmemi_cuda_fn_table *table;
} nvshmemt_libfabric_state_t;

typedef struct {
    struct fid_mr *mr;
    uint64_t key;
    void *local_desc;
} nvshmemt_libfabric_mem_handle_ep_t;

typedef struct {
    nvshmemt_libfabric_mem_handle_ep_t hdls[2];
} nvshmemt_libfabric_mem_handle_t;
