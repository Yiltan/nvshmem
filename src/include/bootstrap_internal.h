/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef BOOTSTRAP_INTERNAL_H
#define BOOTSTRAP_INTERNAL_H

#include <cstddef>
#include "nvshmem_bootstrap.h"

#define MAX_LENGTH_ERROR_STRING 128

enum { BOOTSTRAP_MPI, BOOTSTRAP_SHMEM, BOOTSTRAP_PMI, BOOTSTRAP_PLUGIN };

typedef struct bootstrap_attr {
    bootstrap_attr() : initialize_shmem(0), mpi_comm(NULL) {}
    int initialize_shmem;
    void *mpi_comm;
    void *meta_data;
} bootstrap_attr_t;

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle);
void bootstrap_finalize();

int bootstrap_pmi_init(bootstrap_handle_t *handle);
int bootstrap_pmi2_init(bootstrap_handle_t *handle);
int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle);
int bootstrap_loader_finalize(bootstrap_handle_t *handle);

#endif
