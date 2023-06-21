/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <assert.h>
#include "nvshmemx_error.h"
#include "util.h"
#include "topo.h"
#include "transport.h"
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>

int nvshmemi_transport_show_info(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    for (int i = 0; i < state->num_initialized_transports; ++i) {
        for (size_t j = 0; j < state->handles.size(); j++) {
            status = transports[i]->host_ops.show_info(state->handles[j].data(), i,
                                                       state->num_initialized_transports,
                                                       state->npes, state->mype);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport show info failed \n");
        }
    }
out:
    return status;
}

int nvshmemi_transport_init(nvshmemi_state_t *state) {
    int status = 0;
    int index = 0;
    int transport_skipped;
    nvshmem_transport_t *transports = NULL;
    nvshmemi_transport_init_fn init_fn;
    const int transport_object_file_len = 100;
    char transport_object_file[transport_object_file_len];
    void *transport_lib = NULL;
    bool transport_selected = false;
    nvshmem_local_buf_cache_t *tmp_cache_ptr = NULL;

    status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, done,
                          "Unable to allocate transport mem cache.\n");

    if (!state->transports)
        state->transports =
            (nvshmem_transport_t *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(nvshmem_transport_t));

    transports = (nvshmem_transport_t *)state->transports;

    status = nvshmemt_p2p_init(&transports[index]);
    if (!status) {
        transports[index]->boot_handle = &nvshmemi_boot_handle;
        transports[index]->heap_base = nvshmemi_state->heap_base;
        transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
        transports[index]->index = index;
        transports[index]->log2_cumem_granularity = log2_cumem_granularity;
        transports[index]->cache_handle = tmp_cache_ptr;
        tmp_cache_ptr = NULL;
        if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
        index++;
    } else {
        INFO(NVSHMEM_INIT, "init failed for transport: P2P");
        status = 0;
    }

#ifdef NVSHMEM_IBRC_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, IB_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "IBRC transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ibrc.so.1");
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#ifdef NVSHMEM_UCX_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, UCX_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "UCX transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ucx.so.1");
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#ifdef NVSHMEM_IBDEVX_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, DEVX_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "IBDEVX transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ibdevx.so.1");
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#ifdef NVSHMEM_LIBFABRIC_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, LIBFABRIC_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "IBRC transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_libfabric.so.1");
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

transport_init:

    if (!transport_selected) {
        goto out;
    }

    status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, done,
                          "Unable to allocate transport mem cache.\n");

    transport_lib = dlopen(transport_object_file, RTLD_NOW);
    if (transport_lib == NULL) {
        WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
        goto out;
    }

    init_fn = (nvshmemi_transport_init_fn)dlsym(transport_lib, "nvshmemt_init");
    if (!init_fn) {
        dlclose(transport_lib);
        WARN("Unable to get info from %s transport.\n", transport_object_file);
        goto out;
    }

    status = init_fn(&transports[index], nvshmemi_cuda_syms, NVSHMEM_TRANSPORT_INTERFACE_VERSION);
    if (!status) {
        assert(transports[index]->api_version == NVSHMEM_TRANSPORT_INTERFACE_VERSION);
        transports[index]->boot_handle = &nvshmemi_boot_handle;
        transports[index]->heap_base = nvshmemi_state->heap_base;
        transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
        transports[index]->index = index;
        transports[index]->log2_cumem_granularity = log2_cumem_granularity;
        transports[index]->my_pe = nvshmemi_state->mype;
        transports[index]->n_pes = nvshmemi_state->npes;
        transports[index]->cache_handle = (void *)tmp_cache_ptr;
        tmp_cache_ptr = NULL;
        if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
        state->atomic_host_endian_min_size = transports[index]->atomic_host_endian_min_size;
        index++;
    } else {
        dlclose(transport_lib);
        WARN("init failed for remote transport: %s", nvshmemi_options.REMOTE_TRANSPORT);
        status = 0;
    }
out:
    if (transport_lib) {
        transport_lib = NULL;
        init_fn = NULL;
    }

#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_options.IB_ENABLE_IBGDA) {
        status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, done,
                              "Unable to allocate transport mem cache.\n");

        transport_lib = dlopen("nvshmem_transport_ibgda.so.1", RTLD_NOW);
        if (transport_lib == NULL) {
            WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
            goto done;
        }

        init_fn = (nvshmemi_transport_init_fn)dlsym(transport_lib, "nvshmemt_init");
        if (!init_fn) {
            dlclose(transport_lib);
            WARN("Unable to get info from %s transport.\n", transport_object_file);
            goto done;
        }
        status =
            init_fn(&transports[index], nvshmemi_cuda_syms, NVSHMEM_TRANSPORT_INTERFACE_VERSION);
        if (!status) {
            assert(transports[index]->api_version == NVSHMEM_TRANSPORT_INTERFACE_VERSION);
            transports[index]->boot_handle = &nvshmemi_boot_handle;
            transports[index]->heap_base = nvshmemi_state->heap_base;
            transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
            transports[index]->index = index;
            transports[index]->log2_cumem_granularity = log2_cumem_granularity;
            transports[index]->my_pe = nvshmemi_state->mype;
            transports[index]->n_pes = nvshmemi_state->npes;
            transports[index]->cache_handle = (void *)tmp_cache_ptr;
            tmp_cache_ptr = NULL;
            nvshmemx_gic_get_device_state(&transports[index]->type_specific_shared_state);
            if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
            state->atomic_host_endian_min_size = transports[index]->atomic_host_endian_min_size;
            index++;
        } else {
            WARN("init failed for transport: IBGDA");
            dlclose(transport_lib);
            status = 0;
        }
    } else {
        INFO(NVSHMEM_INIT, "IBGDA Disabled by the environment.");
    }
#endif
done:
    if (transport_lib) {
        transport_lib = NULL;
        init_fn = NULL;
    }

    state->num_initialized_transports = index;

    return status;
}

int nvshmemi_transport_finalize(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_transport_finalize");
    int status = 0;
    nvshmem_transport_t *transports = NULL;
    ;

    if (!state->transports) return 0;

    transports = (nvshmem_transport_t *)state->transports;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (transports[i]->is_successfully_initialized) {
            if (transports[i]->cache_handle) {
                nvshmemi_local_mem_cache_fini(
                    transports[i], (nvshmem_local_buf_cache_t *)transports[i]->cache_handle);
            }
#ifdef NVSHMEM_IBGDA_SUPPORT
            if (transports[i]->type == NVSHMEM_TRANSPORT_LIB_CODE_IBGDA) {
                nvshmemi_device_state.gic_is_initialized = false;
                nvshmemi_gic_update_device_state();
                nvshmemi_set_device_state(&nvshmemi_device_state);
            }
#endif
            status = transports[i]->host_ops.finalize(transports[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport finalize failed \n");
        }
    }
out:
    return status;
}

int nvshmemi_setup_connections(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_transport_t tcurr;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        int selected_device;
        // assumes symmetry of transport list at all PEs
        if ((state->transport_bitmap) & (1 << i)) {
            tcurr = transports[i];
            if (!(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED)) continue;
            if (tcurr->n_devices <= 1) {
                /* return the index of the first available device.
                 * -1 if no devices found.
                 */
                selected_device = tcurr->n_devices - 1;
            } else if (nvshmemi_options.ENABLE_NIC_PE_MAPPING) {
                selected_device = nvshmemi_state->mype_node % tcurr->n_devices;
                INFO(NVSHMEM_INIT, "NVSHMEM_ENABLE_NIC_PE_MAPPING = 1, setting dev_id = %d",
                     selected_device);
            } else {
                nvshmemi_get_device_by_distance(&selected_device, tcurr);
                INFO(NVSHMEM_INIT, "NVSHMEM_ENABLE_NIC_PE_MAPPING = 0, setting dev_id = %d",
                     selected_device);
            }
            status = tcurr->host_ops.connect_endpoints(tcurr, selected_device);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "endpoint connection failed \n");
            status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "barrier failed \n");
#ifdef NVSHMEM_IBGDA_SUPPORT
            if (tcurr->type == NVSHMEM_TRANSPORT_LIB_CODE_IBGDA) {
                nvshmemi_device_state.gic_is_initialized = true;
                status = nvshmemi_gic_update_device_state();

                if (status) {
                    nvshmemi_device_state.gic_is_initialized = false;
                    NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                       "setting IBGDA state failed \n");
                }

                status = nvshmemi_set_device_state(&nvshmemi_device_state);
                if (status) {
                    nvshmemi_device_state.gic_is_initialized = false;
                    NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                       "setting IBGDA state failed \n");
                }
            }
#endif
        }
    }

out:
    return status;
}
