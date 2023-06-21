/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <nvml.h>
#include <string.h>
#include <assert.h>
#include "transport.h"
#include "nvshmemx_error.h"
#include "p2p.h"

int nvshmemt_p2p_init(nvshmem_transport_t *transport);

int nvshmemt_p2p_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id, int transport_count,
                           int npes, int mype) {
    /*XXX : not implemented*/
    return 0;
}

int nvshmemt_p2p_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                nvshmem_transport_t transport) {
    int status = 0;
    int found = 0;
    int p2p_connected = 0;
    CUdevice peer_cudev;
    int peer_devid;
    transport_p2p_state_t *p2p_state = (transport_p2p_state_t *)transport->state;
    int atomics_supported = 0;
    char remote_pcie_bus_id[NVSHMEM_PCIE_DBF_BUFFER_LEN];

    nvmlReturn_t nvml_status;
    nvmlDevice_t remote_device;
    nvmlDevice_t local_device;
    nvmlGpuP2PStatus_t stat;

    if (nvshmemi_options.DISABLE_P2P) {
        INFO(NVSHMEM_INIT, "P2P disabled by user through environment.");
        *access = 0;
        goto out;
    }

    INFO(NVSHMEM_TRANSPORT,
         "[%p] ndev %d pcie_devid %x cudevice %x peer host hash %lx p2p host hash %lx", p2p_state,
         p2p_state->ndev, peer_info->pcie_id.dev_id, p2p_state->cudevice, peer_info->hostHash,
         p2p_state->hostHash);
    if (peer_info->hostHash != p2p_state->hostHash) {
        *access = 0;
        goto out;
    }

    /*find device with the given pcie id*/
    for (int j = 0; j < p2p_state->ndev; j++) {
        if ((p2p_state->pcie_ids[j].dev_id == peer_info->pcie_id.dev_id) &&
            (p2p_state->pcie_ids[j].bus_id == peer_info->pcie_id.bus_id) &&
            (p2p_state->pcie_ids[j].domain_id == peer_info->pcie_id.domain_id)) {
            peer_cudev = p2p_state->cudev[j];
            peer_devid = p2p_state->devid[j];
            found = 1;
            break;
        }
    }

    /* In the case where we don't have access to the GPU directly,
     * and we aren't using VMM, look using NVML.
     */
    if (!found) {
        if (nvshmemi_cuda_driver_version >= 12000 || !nvshmemi_use_cuda_vmm) {
            status = snprintf(remote_pcie_bus_id, NVSHMEM_PCIE_DBF_BUFFER_LEN, "%x:%x:%x.0",
                              peer_info->pcie_id.domain_id, peer_info->pcie_id.bus_id,
                              peer_info->pcie_id.dev_id);
            if (status < 0 || status > NVSHMEM_PCIE_DBF_BUFFER_LEN) {
                INFO(NVSHMEM_TRANSPORT, "Unable to prepare buffer for NVML device detection.\n");
                status = 0;
                goto out;
            }

            status = 0;
            nvml_status = nvmlDeviceGetHandleByPciBusId(remote_pcie_bus_id, &remote_device);
            if (nvml_status != NVML_SUCCESS) {
                INFO(NVSHMEM_TRANSPORT, "Unable to dereference device by UUID using NVML.\n");
                goto out;
            }
            nvml_status = nvmlDeviceGetHandleByPciBusId(p2p_state->pcie_bdf, &local_device);
            if (nvml_status != NVML_SUCCESS) {
                INFO(NVSHMEM_TRANSPORT, "Unable to dereference device by UUID using NVML.\n");
                goto out;
            }
            nvml_status = nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                 NVML_P2P_CAPS_INDEX_READ, &stat);
            if (nvml_status != NVML_SUCCESS) {
                *access = 0;
                INFO(
                    NVSHMEM_TRANSPORT,
                    "Unable to get read status using NVML. Disabling P2P communication for pe %d\n",
                    peer_info->pe);
                goto out;
            } else if (stat == NVML_P2P_STATUS_OK) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD;
            }
            nvml_status = nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                 NVML_P2P_CAPS_INDEX_WRITE, &stat);
            if (nvml_status != NVML_SUCCESS) {
                *access = 0;
                INFO(NVSHMEM_TRANSPORT,
                     "Unable to get write status using NVML. Disabling P2P communication for pe "
                     "%d\n",
                     peer_info->pe);
                goto out;
            } else if (stat == NVML_P2P_STATUS_OK) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST;
            }
            nvml_status = nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                 NVML_P2P_CAPS_INDEX_ATOMICS, &stat);
            if (nvml_status != NVML_SUCCESS) {
                INFO(NVSHMEM_TRANSPORT, "Unable to get atomic status using NVML.\n");
            } else if (stat == NVML_P2P_STATUS_OK) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
            }
            goto out;
        } else {
            /* In the case of CUDA VMM, we can't export a memory handle so LD/ST is also not
             * available. */
            WARN(
                "Some CUDA devices are not visible,\n"
                "likely hidden by CUDA_VISIBLE_DEVICES. Using a network transport to reach these.\n"
                "Disabling VMM usage (dynamic heap) by setting NVSHMEM_DISABLE_CUDA_VMM=1 could "
                "provide better performance.");
            goto out;
        }
    }

    if (peer_cudev == p2p_state->cudevice) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        goto out;
    }

    // use CanAccessPeer if device is visible
    status = cudaDeviceCanAccessPeer(&p2p_connected, p2p_state->device_id, peer_devid);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaDeviceCanAccessPeer failed \n");

    if (p2p_connected) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD;
        status = cudaDeviceGetP2PAttribute(&atomics_supported, cudaDevP2PAttrNativeAtomicSupported,
                                           p2p_state->device_id, peer_devid);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaDeviceGetP2PAttribute failed \n");
        if (atomics_supported) {
            *access |= NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        }
    }

out:
    return status;
}

int nvshmemt_p2p_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                nvshmem_transport_t transport, bool local_only) {
    int status = 0;

    if (local_only) {
        goto out;
    }
#if CUDA_VERSION >= 11000
    if (nvshmemi_use_cuda_vmm) {
        CUmemGenericAllocationHandle *handle_in =
            reinterpret_cast<CUmemGenericAllocationHandle *>(mem_handle_in);
        static_assert(sizeof(CUmemGenericAllocationHandle) <= NVSHMEM_MEM_HANDLE_SIZE,
                      "sizeof(CUmemGenericAllocationHandle) <= NVSHMEM_MEM_HANDLE_SIZE");
        INFO(NVSHMEM_TRANSPORT, "calling cuMemExportToShareableHandle on buf: %p size: %d", buf,
             length);
        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemExportToShareableHandle((void *)mem_handle, *handle_in,
                                                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemExportToShareableHandle failed \n");
    } else
#endif
    {
        cudaIpcMemHandle_t *ipc_handle = (cudaIpcMemHandle_t *)mem_handle;

        assert(sizeof(cudaIpcMemHandle_t) <= NVSHMEM_MEM_HANDLE_SIZE);

        INFO(NVSHMEM_TRANSPORT, "calling cuIpcGetMemHandle on buf: %p size: %zu", buf, length);

        status = cudaIpcGetMemHandle(ipc_handle, buf);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                              "cudaIpcGetMemHandle failed \n");
    }
out:
    return status;
}

int nvshmemt_p2p_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    // it is a noop
    return 0;
}

int nvshmemt_p2p_map(void **buf, size_t size, nvshmem_mem_handle_t *mem_handle) {
    int status = 0;
#if CUDA_VERSION >= 11000
    if (nvshmemi_use_cuda_vmm) {
        CUmemGenericAllocationHandle peer_handle;
        CUmemAccessDesc access;
        CUdevice gpu_device_id;

        status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&gpu_device_id));
        if (status != CUDA_SUCCESS) {
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
        int fd = *(int *)mem_handle;
        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemImportFromShareableHandle(&peer_handle, (void *)(uintptr_t)fd,
                                                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemImportFromShareableHandle failed state->device_id : %d \n",
                              gpu_device_id);

        status = CUPFN(nvshmemi_cuda_syms, cuMemMap((CUdeviceptr)*buf, size, 0, peer_handle, 0));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemMap failed to map %ld bytes handle at address: %p\n", size,
                              *buf);
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = gpu_device_id;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        status = CUPFN(nvshmemi_cuda_syms, cuMemSetAccess((CUdeviceptr)*buf, size,
                                                          (const CUmemAccessDesc *)&access, 1));
    } else
#endif
    {
        cudaIpcMemHandle_t *ipc_handle = (cudaIpcMemHandle_t *)mem_handle;

        status = cudaIpcOpenMemHandle(buf, *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                              "cudaIpcOpenMemHandle failed with error %d \n", status);
    }
out:
    return status;
}

int nvshmemt_p2p_unmap(void *buf, size_t size) {
    int status = 0;

#if CUDA_VERSION >= 11000
    if (nvshmemi_use_cuda_vmm) {
        status = CUPFN(nvshmemi_cuda_syms, cuMemUnmap((CUdeviceptr)buf, size));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                              "cuMemUnmap failed with error %d \n", status);
    } else
#endif
    {
        status = cudaIpcCloseMemHandle(buf);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                              "cudaIpcCloseMemHandle failed with error %d \n", status);
    }
out:
    return status;
}

int nvshmemt_p2p_finalize(nvshmem_transport_t transport) {
    int status = 0;
    nvmlReturn_t nvml_status;

    if (!transport) return 0;

    if (transport->state) {
        transport_p2p_state_t *p2p_state = (transport_p2p_state_t *)transport->state;

        free(p2p_state->cudev);

        free(p2p_state->pcie_ids);

        free(p2p_state);
    }

    nvml_status = nvmlShutdown();
    if (nvml_status != NVML_SUCCESS) {
        INFO(NVSHMEM_TRANSPORT, "Unable to stop nvml library in NVSHMEM.");
    }

    free(transport);

    return status;
}

int nvshmemt_p2p_init(nvshmem_transport_t *t) {
    int status = 0;
    nvmlReturn_t nvml_status;
    struct nvshmem_transport *transport;
    transport_p2p_state_t *p2p_state;

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p transport allocation failed \n");
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized =
        false; /* set it to true after everything has been successfully initialized */

    p2p_state = (transport_p2p_state_t *)calloc(1, sizeof(transport_p2p_state_t));
    NVSHMEMI_NULL_ERROR_JMP(p2p_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&p2p_state->cudevice));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuCtxGetDevice failed \n");

    p2p_state->hostHash = getHostHash();

    status = cudaGetDeviceCount(&p2p_state->ndev);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDeviceCount failed \n");

    p2p_state->cudev = (CUdevice *)malloc(sizeof(CUdevice) * p2p_state->ndev);
    NVSHMEMI_NULL_ERROR_JMP(p2p_state->cudev, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p dev array allocation failed \n");

    p2p_state->devid = (int *)malloc(sizeof(int) * p2p_state->ndev);
    NVSHMEMI_NULL_ERROR_JMP(p2p_state->devid, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p dev array allocation failed \n");

    p2p_state->pcie_ids = (pcie_id_t *)malloc(sizeof(pcie_id_t) * p2p_state->ndev);
    NVSHMEMI_NULL_ERROR_JMP(p2p_state->pcie_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p pcie_ids array allocation failed \n");

    for (int i = 0; i < p2p_state->ndev; i++) {
        status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&p2p_state->cudev[i], i));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuDeviceGet failed \n");
        p2p_state->devid[i] = i;

        if (p2p_state->cudev[i] == p2p_state->cudevice) {
            p2p_state->device_id = i;
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaGetDeviceProperties failed \n");
            status = snprintf(p2p_state->pcie_bdf, NVSHMEM_PCIE_DBF_BUFFER_LEN, "%x:%x:%x.0",
                              prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            if (status < 0 || status > NVSHMEM_PCIE_DBF_BUFFER_LEN) {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to set device pcie bdf for our local device.\n");
            }
        }

        status = nvshmemi_get_pcie_attrs(&p2p_state->pcie_ids[i], p2p_state->cudev[i]);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemi_get_pcie_attrs failed \n");
    }

    /* start NVML Library */
    nvml_status = nvmlInit();
    if (nvml_status != NVML_SUCCESS) {
        INFO(NVSHMEM_INIT, "Unable to open nvml. Some topology detection will be disabled.");
    }

    transport->host_ops.can_reach_peer = nvshmemt_p2p_can_reach_peer;
    transport->host_ops.get_mem_handle = nvshmemt_p2p_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_p2p_release_mem_handle;
    transport->host_ops.map = nvshmemt_p2p_map;
    transport->host_ops.unmap = nvshmemt_p2p_unmap;
    transport->host_ops.finalize = nvshmemt_p2p_finalize;
    transport->host_ops.show_info = nvshmemt_p2p_show_info;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS;
    transport->state = p2p_state;
    transport->is_successfully_initialized = true;
    transport->no_proxy = true;

    *t = transport;

out:
    if (status) {
        if (transport) {
            free(transport);
            if (p2p_state) {
                if (p2p_state->cudev) {
                    free(p2p_state->cudev);
                }
                if (p2p_state->pcie_ids) {
                    free(p2p_state->pcie_ids);
                }
                free(p2p_state);
            }
        }
    }
    return status;
}
