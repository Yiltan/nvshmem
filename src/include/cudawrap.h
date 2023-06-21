/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NVSHMEM_CUDAWRAP_H
#define NVSHMEM_CUDAWRAP_H

#include <cuda.h>

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#else
typedef CUresult(CUDAAPI *PFN_cuInit_v2000)(unsigned int Flags);
typedef CUresult(CUDAAPI *PFN_cuGetProcAddress_v11030)(const char *symbol, void **pfn,
                                                       int driverVersion, cuuint64_t flags);
typedef CUresult(CUDAAPI *PFN_cuDeviceGetAttribute_v2000)(int *pi, CUdevice_attribute attrib,
                                                          CUdevice dev);
typedef CUresult(CUDAAPI *PFN_cuPointerSetAttribute_v6000)(const void *value,
                                                           CUpointer_attribute attribute,
                                                           CUdeviceptr ptr);
typedef CUresult(CUDAAPI *PFN_cuGetErrorString_v6000)(CUresult error, const char **pStr);
typedef CUresult(CUDAAPI *PFN_cuGetErrorName_v6000)(CUresult error, const char **pStr);
typedef CUresult(CUDAAPI *PFN_cuDeviceGet_v2000)(CUdevice *device, int ordinal);
typedef CUresult(CUDAAPI *PFN_cuCtxSetCurrent_v4000)(CUcontext ctx);
typedef CUresult(CUDAAPI *PFN_cuCtxGetDevice_v2000)(CUdevice *device);
typedef CUresult(CUDAAPI *PFN_cuCtxGetCurrent_v4000)(CUcontext *pctx);
typedef CUresult(CUDAAPI *PFN_cuDevicePrimaryCtxRetain_v7000)(CUcontext *pctx, CUdevice dev);
typedef CUresult(CUDAAPI *PFN_cuCtxSynchronize_v2000)();
typedef CUresult(CUDAAPI *PFN_cuModuleGetGlobal_v3020)(CUdeviceptr *dptr, size_t *bytes,
                                                       CUmodule hmod, const char *name);
#if CUDA_VERSION >= 11000
typedef CUresult(CUDAAPI *PFN_cuMemCreate_v10020)(CUmemGenericAllocationHandle *handle, size_t size,
                                                  const CUmemAllocationProp *prop,
                                                  unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuMemGetAllocationGranularity_v10020)(
    size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
typedef CUresult(CUDAAPI *PFN_cuMemAddressReserve_v10020)(CUdeviceptr *ptr, size_t size,
                                                          size_t alignment, CUdeviceptr addr,
                                                          unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuMemExportToShareableHandle_v10020)(
    void *shareableHandle, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handleType, unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuMemImportFromShareableHandle_v10020)(
    CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
typedef CUresult(CUDAAPI *PFN_cuMemMap_v10020)(CUdeviceptr ptr, size_t size, size_t offset,
                                               CUmemGenericAllocationHandle handle,
                                               unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuMemRelease_v10020)(CUmemGenericAllocationHandle handle);
typedef CUresult(CUDAAPI *PFN_cuMemSetAccess_v10020)(CUdeviceptr ptr, size_t size,
                                                     const CUmemAccessDesc *desc, size_t count);
typedef CUresult(CUDAAPI *PFN_cuMemUnmap_v10020)(CUdeviceptr ptr, size_t size);
typedef CUresult(CUDAAPI *PFN_cuMemGetAccess_v10020)(unsigned long long *flags,
                                                     const CUmemLocation *location,
                                                     CUdeviceptr ptr);
#endif
#endif

#define DEFINE_SYM(symbol, version) PFN_##symbol##_v##version pfn_##symbol;
struct nvshmemi_cuda_fn_table {
    DEFINE_SYM(cuCtxGetDevice, 2000)
    DEFINE_SYM(cuCtxSynchronize, 2000)
    DEFINE_SYM(cuDeviceGet, 2000)
    DEFINE_SYM(cuDeviceGetAttribute, 2000)
    DEFINE_SYM(cuPointerSetAttribute, 6000)
    DEFINE_SYM(cuModuleGetGlobal, 3020)
    DEFINE_SYM(cuGetErrorString, 6000)
    DEFINE_SYM(cuGetErrorName, 6000)
    DEFINE_SYM(cuCtxSetCurrent, 4000)
    DEFINE_SYM(cuDevicePrimaryCtxRetain, 7000)
    DEFINE_SYM(cuCtxGetCurrent, 4000)
#if CUDA_VERSION >= 11070
    DEFINE_SYM(cuMemGetHandleForAddressRange, 11070)  // DMA-BUF support
#endif
#if CUDA_VERSION >= 11000
    DEFINE_SYM(cuMemCreate, 10020)
    DEFINE_SYM(cuMemAddressReserve, 10020)
    DEFINE_SYM(cuMemMap, 10020)
    DEFINE_SYM(cuMemGetAllocationGranularity, 10020)
    DEFINE_SYM(cuMemImportFromShareableHandle, 10020)
    DEFINE_SYM(cuMemExportToShareableHandle, 10020)
    DEFINE_SYM(cuMemRelease, 10020)
    DEFINE_SYM(cuMemSetAccess, 10020)
    DEFINE_SYM(cuMemUnmap, 10020)
#endif

    /* CUDA Driver functions loaded with dlsym() */
    DEFINE_SYM(cuInit, 2000)
    DEFINE_SYM(cuGetProcAddress, 11030)
};
#undef DEFINE_SYM

#define CUPFN(table, symbol) table->pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECKNORETURN(table, cmd)                          \
    do {                                                     \
        CUresult err = table->pfn_##cmd;                     \
        if (err != CUDA_SUCCESS) {                           \
            const char *errStr;                              \
            (void)table->pfn_cuGetErrorString(err, &errStr); \
            fprintf(stderr, "Cuda failure '%s'", errStr);    \
        }                                                    \
        assert(err == CUDA_SUCCESS);                         \
    } while (false)

// Check CUDA PFN driver calls
#define CUCHECK(table, cmd)                                  \
    do {                                                     \
        CUresult err = table->pfn_##cmd;                     \
        if (err != CUDA_SUCCESS) {                           \
            const char *errStr;                              \
            (void)table->pfn_cuGetErrorString(err, &errStr); \
            fprintf(stderr, "Cuda failure '%s'", errStr);    \
            return NVSHMEMX_ERROR_INTERNAL;                  \
        }                                                    \
    } while (false)

#define CUCHECKGOTO(table, cmd, res, label)                  \
    do {                                                     \
        CUresult err = table->pfn_##cmd;                     \
        if (err != CUDA_SUCCESS) {                           \
            const char *errStr;                              \
            (void)table->pfn_cuGetErrorString(err, &errStr); \
            fprintf(stderr, "Cuda failure '%s'", errStr);    \
            res = NVSHMEMX_ERROR_INTERNAL;                   \
            goto label;                                      \
        }                                                    \
    } while (false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(table, cmd)                                                   \
    do {                                                                            \
        CUresult err = table->pfn_##cmd;                                            \
        if (err != CUDA_SUCCESS) {                                                  \
            const char *errStr;                                                     \
            (void)table->pfn_cuGetErrorString(err, &errStr);                        \
            fprintf(stderr, "%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr); \
        }                                                                           \
    } while (false)

#define CUCHECKTHREAD(table, cmd, args)                                             \
    do {                                                                            \
        CUresult err = table->pfn_##cmd;                                            \
        if (err != CUDA_SUCCESS) {                                                  \
            fprintf(stderr, "%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
            args->ret = NVSHMEMX_ERROR_INTERNAL;                                    \
            return args;                                                            \
        }                                                                           \
    } while (0)

#endif
