/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "error_codes_internal.h"
#include "debug.h"
#include "cudawrap.h"

#include <dlfcn.h>

static enum {
    cudaUninitialized,
    cudaInitializing,
    cudaInitialized,
    cudaError
} cudaState = cudaUninitialized;

static void *cudaLib;
static int cudaDriverVersion;

static int cudaPfnFuncLoader(struct nvshmemi_cuda_fn_table *table) {
    CUresult res;

#define LOAD_SYM(table, symbol, version, sym_suffix, ignore)                                       \
    do {                                                                                           \
        bool not_found = false;                                                                    \
        if (table->pfn_cuGetProcAddress) {                                                         \
            res =                                                                                  \
                table->pfn_cuGetProcAddress(#symbol, (void **)(&table->pfn_##symbol), version, 0); \
            if (res != 0) not_found = true;                                                        \
        } else {                                                                                   \
            table->pfn_##symbol = (PFN_##symbol##_v##version)dlsym(cudaLib, #symbol #sym_suffix);  \
            if (table->pfn_##symbol == NULL) not_found = true;                                     \
        }                                                                                          \
        if (not_found) {                                                                           \
            if (!ignore) {                                                                         \
                WARN("Retrieve %s version %d failed", #symbol #sym_suffix, cudaDriverVersion);     \
                return NVSHMEMI_SYSTEM_ERROR;                                                      \
            }                                                                                      \
        }                                                                                          \
    } while (0)

    LOAD_SYM(table, cuCtxGetDevice, 2000, , 0);
    LOAD_SYM(table, cuCtxSynchronize, 2000, , 0);
    LOAD_SYM(table, cuDeviceGet, 2000, , 0);
    LOAD_SYM(table, cuDeviceGetAttribute, 2000, , 0);
    LOAD_SYM(table, cuPointerSetAttribute, 6000, , 0);
    LOAD_SYM(table, cuModuleGetGlobal, 3020, _v2, 0);
    LOAD_SYM(table, cuGetErrorString, 6000, , 0);
    LOAD_SYM(table, cuGetErrorName, 6000, , 0);
    LOAD_SYM(table, cuCtxSetCurrent, 4000, , 0);
    LOAD_SYM(table, cuDevicePrimaryCtxRetain, 7000, , 0);
    LOAD_SYM(table, cuCtxGetCurrent, 4000, , 0);
#if CUDA_VERSION >= 11070
    LOAD_SYM(table, cuMemGetHandleForAddressRange, 11070, , 1);  // DMA-BUF support
#endif
#if CUDA_VERSION >= 11000
    LOAD_SYM(table, cuMemCreate, 10020, , 1);
    LOAD_SYM(table, cuMemMap, 10020, , 1);
    LOAD_SYM(table, cuMemAddressReserve, 10020, , 1);
    LOAD_SYM(table, cuMemGetAllocationGranularity, 10020, , 1);
    LOAD_SYM(table, cuMemImportFromShareableHandle, 10020, , 1);
    LOAD_SYM(table, cuMemExportToShareableHandle, 10020, , 1);
    LOAD_SYM(table, cuMemRelease, 10020, , 1);
    LOAD_SYM(table, cuMemSetAccess, 10020, , 1);
    LOAD_SYM(table, cuMemUnmap, 10020, , 1);
#endif
    return NVSHMEMI_SUCCESS;
}

int nvshmemi_cuda_library_init(struct nvshmemi_cuda_fn_table *table) {
    cudaError_t cuda_err;

    if (cudaState == cudaInitialized) return NVSHMEMI_SUCCESS;
    if (cudaState == cudaError) return NVSHMEMI_SYSTEM_ERROR;

    /*
     * Load CUDA driver library
     */
    char path[1024];
    char *nvshmemCudaPath = getenv("NVSHMEM_CUDA_PATH");
    if (nvshmemCudaPath == NULL)
        snprintf(path, 1024, "%s", "libcuda.so");
    else
        snprintf(path, 1024, "%s/%s", nvshmemCudaPath, "libcuda.so");

    cudaLib = dlopen(path, RTLD_LAZY);
    if (cudaLib == NULL) {
        WARN("Failed to find CUDA library in %s (NVSHMEM_CUDA_PATH=%s)", nvshmemCudaPath,
             nvshmemCudaPath);
        goto error;
    }

    /*
     * Load initial CUDA functions
     */

    table->pfn_cuInit = (PFN_cuInit_v2000)dlsym(cudaLib, "cuInit");
    if (table->pfn_cuInit == NULL) {
        WARN("Failed to load CUDA missing symbol cuInit");
        goto error;
    }

    cuda_err = cudaDriverGetVersion(&cudaDriverVersion);
    if (cuda_err != 0) {
        WARN("cudaDriverGetVersion failed with %d", cuda_err);
        goto error;
    }
    INFO(NVSHMEM_INIT, "cudaDriverVersion %d", cudaDriverVersion);

    table->pfn_cuGetProcAddress = (PFN_cuGetProcAddress_v11030)dlsym(cudaLib, "cuGetProcAddress");

    /*
     * Required to initialize the CUDA Driver.
     * Multiple calls of cuInit() will return immediately
     * without making any relevant change
     */
    table->pfn_cuInit(0);

    if (cudaPfnFuncLoader(table)) {
        WARN("CUDA some PFN functions not found in the library");
        goto error;
    }

    cudaState = cudaInitialized;
    return NVSHMEMI_SUCCESS;

error:
    cudaState = cudaError;
    return NVSHMEMI_SYSTEM_ERROR;
}
