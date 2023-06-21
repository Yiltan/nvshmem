/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"
#include <dlfcn.h>
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"
struct nccl_function_table nccl_ftable;
#endif /* NVSHMEM_USE_NCCL */

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

int nvshmemi_use_nccl = 0;
int nccl_version;

int nvshmemi_coll_common_cpu_read_env() {
    int status = 0;
    nvshmemi_device_state.fcollect_ll_threshold = nvshmemi_options.FCOLLECT_LL_THRESHOLD;
    return status;
}

int nvshmemi_coll_common_cpu_init() {
    int status = 0;
#ifdef NVSHMEM_USE_NCCL
    void *nccl_handle = NULL;
    int nccl_build_version;
    int nccl_major;
    int nccl_build_major;
#endif

    status = nvshmemi_coll_common_cpu_read_env();
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

#ifdef NVSHMEM_USE_NCCL
    nvshmemi_use_nccl = 1;
    assert(NCCL_VERSION_CODE >= 2000);
    if (nvshmemi_options.DISABLE_NCCL) {
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }

    nccl_handle = dlopen("libnccl.so.2", RTLD_LAZY);
    if (!nccl_handle) {
        NVSHMEMI_WARN_PRINT("NCCL library not found...\n");
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }

    nccl_build_version = NCCL_VERSION_CODE;
    LOAD_SYM(nccl_handle, "ncclGetVersion", nccl_ftable.GetVersion);
    nccl_ftable.GetVersion(&nccl_version);
    if (nccl_version > 10000) {
        nccl_major = nccl_version / 10000;
    } else {
        nccl_major = nccl_version / 1000;
    }
    if (nccl_build_version > 10000) {
        nccl_build_major = nccl_build_version / 10000;
    } else {
        nccl_build_major = nccl_build_version / 1000;
    }
    if (nccl_major != nccl_build_major) {
        NVSHMEMI_WARN_PRINT(
            "NCCL library major version (%d) is different than the"
            " version (%d) with which NVSHMEM was built, skipping use...\n",
            nccl_major, nccl_build_major);
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }
    LOAD_SYM(nccl_handle, "ncclGetUniqueId", nccl_ftable.GetUniqueId);
    LOAD_SYM(nccl_handle, "ncclCommInitRank", nccl_ftable.CommInitRank);
    LOAD_SYM(nccl_handle, "ncclCommDestroy", nccl_ftable.CommDestroy);
    LOAD_SYM(nccl_handle, "ncclAllReduce", nccl_ftable.AllReduce);
    LOAD_SYM(nccl_handle, "ncclBroadcast", nccl_ftable.Broadcast);
    LOAD_SYM(nccl_handle, "ncclAllGather", nccl_ftable.AllGather);
    LOAD_SYM(nccl_handle, "ncclGetErrorString", nccl_ftable.GetErrorString);
    LOAD_SYM(nccl_handle, "ncclGroupStart", nccl_ftable.GroupStart);
    LOAD_SYM(nccl_handle, "ncclGroupEnd", nccl_ftable.GroupEnd);
    if (nccl_version >= 2700) {
        LOAD_SYM(nccl_handle, "ncclSend", nccl_ftable.Send);
        LOAD_SYM(nccl_handle, "ncclRecv", nccl_ftable.Recv);
    }

fn_out:
#endif /* NVSHMEM_USE_NCCL */
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_cpu_finalize() {
    int status = 0;

    return status;
}
