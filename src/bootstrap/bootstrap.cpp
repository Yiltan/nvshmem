/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include "util.h"
#include "nvshmemx_error.h"
#include "bootstrap_internal.h"

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;
    const char *plugin_name = NULL;

    switch (mode) {
        case BOOTSTRAP_MPI:
            plugin_name = nvshmemi_options.BOOTSTRAP_MPI_PLUGIN;

            status =
                bootstrap_loader_init(plugin_name, (attr != NULL) ? attr->mpi_comm : NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error\n");
            break;
        case BOOTSTRAP_SHMEM:
            plugin_name = nvshmemi_options.BOOTSTRAP_SHMEM_PLUGIN;

            status = bootstrap_loader_init(plugin_name,
                                           (attr != NULL) ? &attr->initialize_shmem : NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error\n");
            break;
        case BOOTSTRAP_PMI:
            if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMIX") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMIX_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_loader_init returned error\n");
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI-2") == 0 ||
                       strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI2") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMI2_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_pmi_init returned error\n");
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMI_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_pmi_init returned error\n");
            } else {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "bootstrap_pmi_init invalid PMI bootstrap '%s'\n",
                                   nvshmemi_options.BOOTSTRAP_PMI);
            }
            break;
        case BOOTSTRAP_PLUGIN:
            assert(attr == NULL);

            if (!nvshmemi_options.BOOTSTRAP_PLUGIN_provided) {
                NVSHMEMI_ERROR_PRINT(
                    "Plugin bootstrap requires NVSHMEM_BOOTSTRAP_PLUGIN to be set\n");
                status = 1;
                goto out;
            }

            plugin_name = nvshmemi_options.BOOTSTRAP_PLUGIN;

            status = bootstrap_loader_init(nvshmemi_options.BOOTSTRAP_PLUGIN, NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error\n");
            break;
        default:
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap mode selected\n");
    }

out:
    return status;
}

void bootstrap_finalize() {
    int status = NVSHMEMX_SUCCESS;

    if (nvshmemi_is_nvshmem_bootstrapped) {
        status = bootstrap_loader_finalize(&nvshmemi_boot_handle);
        NVSHMEMI_NZ_EXIT(status, "bootstrap finalization returned error\n");
        NVSHMEMU_THREAD_CS_FINALIZE();
    }
}
