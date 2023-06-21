/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <set>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "nvshmem_nvtx.hpp"

#include <stdlib.h>
#include <string.h>
#include "topo.h"
#include "util.h"
#include "cpu_coll.h"
#include "gpu_coll.h"
#include "unistd.h"
#include "debug.h"

static std::set<nvshmemi_state_change_handler_fn_t> state_change_handler_set;
static std::set<void *> registered_ibgda_states;

static void nvshmemi_init_debug(void);
static void nvshmemi_init_msg(void);

struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;
nvshmemi_state_t *nvshmemi_state;
bootstrap_handle_t nvshmemi_boot_handle;
nvshmemi_pe_dist_t nvshmemi_pe_dist;
uint64_t *nvshmemi_host_hashes;
bool nvshmemi_is_nvshmem_bootstrapped = false;
bool nvshmemi_is_nvshmem_initialized = false;
int nvshmemi_init_counter = 0;
nvshmem_options_t nvshmem_options;
int nvshmemi_cuda_driver_version;
bool nvshmemi_use_cuda_vmm = 0;
const char *p_err_str;
int nvshmem_debug_level;
uint64_t nvshmem_debug_mask = NVSHMEM_INIT;  // Default debug sub-system mask is INIT
pthread_mutex_t nvshmem_debug_output_lock;
bool nvshmemi_is_limited_mpg_run = 0;
FILE *nvshmem_debug_file = stdout;
static char shm_name[100];
nvshmemi_version_t nvshmemi_host_lib_version = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};

#ifdef NVSHMEM_TRACE
std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#endif

void (*nvshmemi_check_state_and_init_fn_ptr)();

void *heap_base_array_dptr = NULL;
void *heap_base_actual_array_dptr = NULL;
int nvshmemi_job_connectivity;

nvshmemi_device_state_t nvshmemi_device_state;
bool nvshmemi_is_device_state_initialized = false;

static int nvshmemi_transport_cap_support_rma(int cap) {
    if (cap & (NVSHMEM_TRANSPORT_CAP_CPU_READ | NVSHMEM_TRANSPORT_CAP_CPU_WRITE |
               NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST)) {
        return 1;
    }
    return 0;
}

static int nvshmemi_transport_cap_support_amo(int cap) {
    if (cap & (NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS)) {
        return 1;
    }
    return 0;
}

bool nvshmemi_is_version_compatible(const nvshmemi_version_t version_1,
                                    const nvshmemi_version_t version_2) {
    if (version_1.major == version_2.major && version_1.minor == version_2.minor &&
        version_1.patch == version_2.patch) {
        return 0;
    }
    return 1;
}

int nvshmemi_bootstrap(int flags, nvshmemx_init_attr_t *nvshmem_attr) {
    int status = 0;
    uint64_t myHostHash = 0;
    uint64_t *hostHash = NULL;
    int mype, npes;
    int mype_node = 0, npes_node = 0;
    int num_nodes;

    if (flags & NVSHMEMX_INIT_WITH_MPI_COMM) {
        bootstrap_attr_t boot_attr;
        boot_attr.mpi_comm = nvshmem_attr->mpi_comm;
        status = bootstrap_init(BOOTSTRAP_MPI, &boot_attr, &nvshmemi_boot_handle);
    } else if (flags & NVSHMEMX_INIT_WITH_SHMEM) {
        bootstrap_attr_t boot_attr;
        boot_attr.initialize_shmem = 0;
        status = bootstrap_init(BOOTSTRAP_SHMEM, &boot_attr, &nvshmemi_boot_handle);
    } else {
        /* User called nvshmem_init or supplied no flags to nvshmemx_init_attr */
        if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "PMI") == 0) {
            status = bootstrap_init(BOOTSTRAP_PMI, NULL, &nvshmemi_boot_handle);
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "MPI") == 0) {
            status = bootstrap_init(BOOTSTRAP_MPI, NULL, &nvshmemi_boot_handle);
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "SHMEM") == 0) {
            status = bootstrap_init(BOOTSTRAP_SHMEM, NULL, &nvshmemi_boot_handle);
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "plugin") == 0) {
            status = bootstrap_init(BOOTSTRAP_PLUGIN, NULL, &nvshmemi_boot_handle);
        } else {
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap '%s'\n", nvshmemi_options.BOOTSTRAP);
            status = 1;
        }
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_init failed \n");

    mype = nvshmemi_boot_handle.pg_rank;
    npes = nvshmemi_boot_handle.pg_size;
    myHostHash = getHostHash();
    hostHash = (uint64_t *)malloc(sizeof(uint64_t) * npes);
    status = nvshmemi_boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of host hashes failed \n");
    nvshmemi_host_hashes = (uint64_t *)malloc(sizeof(uint64_t) * npes);
    memcpy(nvshmemi_host_hashes, hostHash, sizeof(uint64_t) * npes);
    for (int i = 0; i < npes; i++) {
        if (nvshmemi_host_hashes[i] == myHostHash) {
            if (i == mype) mype_node = npes_node;
            npes_node++;
        }
    }
    nvshmemi_boot_handle.mype_node = mype_node;
    nvshmemi_boot_handle.npes_node = npes_node;

    // Check for same number of PEs on every node
    // Use myHostHash value to indicate a given PE has been counted already
    // This overwrites the hostHash values, but we don't them past this point

    for (int i = 0; i < npes; i++) {
        // If this host's node hasn't been counted yet, count it
        if (hostHash[i] != myHostHash) {
            const uint64_t peer_hash = hostHash[i];
            int npes_peer_node = 0;
            for (int j = i; j < npes; j++) {
                if (peer_hash == hostHash[j]) {
                    npes_peer_node++;
                    hostHash[j] = myHostHash;
                }
            }
            if (npes_peer_node != npes_node) {
                char hostname[1024];
                nvshmemu_gethostname(hostname, 1024);

                NVSHMEMI_ERROR_JMP(
                    status, NVSHMEMX_ERROR_INTERNAL, out,
                    "NVSHMEM requires the same number of PEs on all nodes (%d PEs on %s)\n",
                    npes_node, hostname);
            }
        }
    }

    nvshmem_nvtx_set_thread_name(mype);

    /* Set pe distribution type. First check for round robin distribution. Then check for block
     * distribution. */
    nvshmemi_pe_dist = NVSHMEMI_PE_DIST_MISC;
    if (npes % npes_node != 0) goto out;
    num_nodes = npes / npes_node;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < npes_node; j++) {
            if (nvshmemi_host_hashes[i * npes_node] != nvshmemi_host_hashes[i * num_nodes + j])
                goto check_roundrobin_dist;
        }
    }
    nvshmemi_pe_dist = NVSHMEMI_PE_DIST_BLOCK;
    INFO(NVSHMEM_INIT, "PE distribution has been identified as NVSHMEMI_PE_DIST_BLOCK");
    goto out;

check_roundrobin_dist:
    for (int i = 0; i < npes_node; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (nvshmemi_host_hashes[j * npes_node] != nvshmemi_host_hashes[i * num_nodes + j])
                goto out;
        }
    }
    nvshmemi_pe_dist = NVSHMEMI_PE_DIST_ROUNDROBIN;
    INFO(NVSHMEM_INIT, "PE distribution has been identified as NVSHMEMI_PE_DIST_ROUNDROBIN");

out:
    nvshmemi_device_state.pe_dist = nvshmemi_pe_dist;
    if (hostHash) free(hostHash);
    return status;
}

int nvshmemi_init_nvshmemi_state(nvshmemi_state_t *state) {
    int status = 0;
    state->mype = nvshmemi_boot_handle.pg_rank;
    state->npes = nvshmemi_boot_handle.pg_size;
    state->mype_node = nvshmemi_boot_handle.mype_node;
    state->npes_node = nvshmemi_boot_handle.npes_node;

    return status;
}

int nvshmemi_get_cucontext(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    CUCHECK(nvshmemi_cuda_syms, cuInit(0));

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&state->cudevice));
    if (status || nvshmemi_options.BOOTSTRAP_TWO_STAGE) {
        if (nvshmemi_options.BOOTSTRAP_TWO_STAGE) {
            TRACE(NVSHMEM_INIT, "Two-stage initialization requested");
            nvshmemi_options.BOOTSTRAP_TWO_STAGE = false;
        } else
            TRACE(NVSHMEM_INIT, "GPU not selected, cuCtxGetDevice failed, err: %d", status);

        status = NVSHMEMX_ERROR_GPU_NOT_SELECTED;
        goto out;
    } else {
        CUresult cres = CUPFN(nvshmemi_cuda_syms, cuCtxSynchronize());
        if (cres) {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->%d(CUDA_ERROR_NOT_INITIALIZED %d) "
                 "my_stream %llu",
                 state->mype, cres, CUDA_ERROR_NOT_INITIALIZED, state->my_stream);
            CUCHECK(nvshmemi_cuda_syms,
                    cuDevicePrimaryCtxRetain(&state->cucontext, state->cudevice));
            CUCHECK(nvshmemi_cuda_syms, cuCtxSetCurrent(state->cucontext));
            INFO(NVSHMEM_INIT, "retained primary context for device: %d", state->cudevice);
        } else {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->CUDA_SUCCESS) my_stream %p",
                 state->mype, state->my_stream);
            CUCHECK(nvshmemi_cuda_syms, cuCtxGetCurrent(&state->cucontext));
            INFO(NVSHMEM_INIT,
                 "in get_cucontext, queried and saved context for device: %d context: %p",
                 state->cudevice, state->cucontext);
        }

        // identify device id
        int count;
        CUdevice curr_device;
        CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&count));

        for (int i = 0; i < count; i++) {
            CUCHECK(nvshmemi_cuda_syms, cuDeviceGet(&curr_device, i));
            if (curr_device == state->cudevice) {
                state->device_id = i;
                break;
            }
        }
        CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_RUNTIME_CHECK(cudaStreamCreateWithPriority(&state->my_stream, cudaStreamNonBlocking,
                                                        greatestPriority));
        INFO(NVSHMEM_INIT,
             "[%d] nvshmemi_get_cucontext->cuCtxGetDevice->%d(CUDA_ERROR_INVALID_CONTEXT %d) "
             "cuStreamCreateWithPriority my_stream %p",
             state->mype, cres, CUDA_ERROR_INVALID_CONTEXT, state->my_stream);
    }
out:
    return status;
}

int nvshmemi_setup_stream_priorities(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_RUNTIME_CHECK(
        cudaStreamCreateWithPriority(&state->my_stream, cudaStreamNonBlocking, greatestPriority));

    return status;
}

int nvshmemi_setup_memory_ordering(nvshmemi_state_t *state, int selected_transport) {
    int status = 0;

    state->fence[selected_transport] = state->transports[selected_transport]->host_ops.fence;
    state->quiet[selected_transport] = state->transports[selected_transport]->host_ops.quiet;

    return status;
}

int nvshmemi_teardown_handles(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_teardown_handles");
    int status = 0;
    free(state->rma);
    free(state->selected_transport_for_rma);
    free(state->amo);
    free(state->selected_transport_for_amo);
    free(state->fence);
    free(state->quiet);
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamDestroy(state->custreams[i]), status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(state->cuevents[i]), status, out);
    }
out:
    return status;
}

static int nvshmemi_setup_nvshmem_handles(nvshmemi_state_t *state) {
    int status = 0;
    int dev_attr = 0;
    /* TODO: We should really check all of these allocations. */
    state->rma = (rma_handle *)calloc(state->npes, sizeof(rma_handle));
    state->amo = (amo_handle *)calloc(state->npes, sizeof(amo_handle));
    state->fence = (fence_handle *)calloc(state->num_initialized_transports, sizeof(fence_handle));
    state->quiet = (quiet_handle *)calloc(state->num_initialized_transports, sizeof(quiet_handle));
    state->selected_transport_for_rma = (int *)calloc(state->npes, sizeof(int));
    state->selected_transport_for_amo = (int *)calloc(state->npes, sizeof(int));
    CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(
        &dev_attr, cudaDevAttrCanUseHostPointerForRegisteredMem, state->device_id));
    state->host_memory_registration_supported =
        dev_attr & cudaDevAttrCanUseHostPointerForRegisteredMem;

    for (int pe = 0; pe < state->npes; pe++) {
        state->rma[pe] = 0;
        state->amo[pe] = 0;
        state->selected_transport_for_rma[pe] = -1;
        state->selected_transport_for_amo[pe] = -1;
    }
    for (int t = 0; t < state->num_initialized_transports; t++) {
        state->fence[t] = 0;
        state->quiet[t] = 0;
    }
    int memory_ordering_initialized = 0;
    int tbitmap;
    for (int i = 0; i < state->npes; i++) {
        bool amo_initialized = false, rma_initialized = false;
        tbitmap = state->transport_bitmap;
        for (int j = 0; j < state->num_initialized_transports; j++) {
            if (!(state->transports[j])) {
                tbitmap >>= 1;
                continue;
            }

            if (tbitmap & 1) {
                if (!rma_initialized &&
                    nvshmemi_transport_cap_support_rma(nvshmemi_state->transports[j]->cap[i])) {
                    state->rma[i] = state->transports[j]->host_ops.rma;
                    rma_initialized = true;
                    state->selected_transport_for_rma[i] = j;
                }

                if (!amo_initialized &&
                    nvshmemi_transport_cap_support_amo(nvshmemi_state->transports[j]->cap[i])) {
                    state->amo[i] = state->transports[j]->host_ops.amo;
                    amo_initialized = true;
                    state->selected_transport_for_amo[i] = j;
                }

                if (((state->selected_transport_for_amo[i] == j) ||
                     (state->selected_transport_for_rma[i] == j)) &&
                    !(memory_ordering_initialized & (1 << j))) {
                    nvshmemi_setup_memory_ordering(state, j);
                    memory_ordering_initialized |= 1 << j;
                }
            }
            tbitmap >>= 1;
        }
    }

    return status;
}

static int nvshmemi_setup_cuda_handles(nvshmemi_state_t *state) {
    int status = 0;
    state->custreams = (cudaStream_t *)malloc(MAX_PEER_STREAMS * sizeof(cudaStream_t));
    state->cuevents = (cudaEvent_t *)malloc(MAX_PEER_STREAMS * sizeof(cudaEvent_t));
    state->active_internal_streams = (bool *)calloc(MAX_PEER_STREAMS, sizeof(bool));
    int leastPriority, greatestPriority;
    CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamCreateWithPriority(
                                    &state->custreams[i], cudaStreamNonBlocking, greatestPriority),
                                status, out);
        CUDA_RUNTIME_CHECK_GOTO(
            cudaEventCreateWithFlags(&state->cuevents[i], cudaEventDisableTiming), status, out);
    }
out:
    return status;
}

static int shared_memory_create(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
    int status = 0;

    info->size = sz;

    info->shm_fd = shm_open(name, O_RDWR | O_CREAT, 0777);
    if (info->shm_fd < 0) {
        INFO(NVSHMEM_INIT, "shm_open failed");
        return errno;
    }

    status = ftruncate(info->shm_fd, sz);
    if (status != 0) {
        INFO(NVSHMEM_INIT, "ftruncate failed");
        return status;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
    if (info->addr == NULL) {
        INFO(NVSHMEM_INIT, "mmap failed");
        return errno;
    }

    return status;
}

int shared_memory_open(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
    int status = 0;
    info->size = sz;
    struct stat stat_shm;

    info->shm_fd = shm_open(name, O_RDWR, 0777);
    if (info->shm_fd < 0) {
        return errno;
    }

    status = fstat(info->shm_fd, &stat_shm);
    if (status != 0) {
        INFO(NVSHMEM_INIT, "fstat failed");
        return status;
    }
    assert(stat_shm.st_size == (intmax_t)sz);

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return status;
}

static void shared_memory_close(nvshmemi_shared_memory_info *info) {
    if (info->addr) {
        munmap(info->addr, info->size);
    }

    shm_unlink(shm_name);
}

static bool trimNewline(char *str) {
    size_t len = strlen(str);
    if (len > 0 && str[len - 1] == '\n') {
        str[len - 1] = '\0';
    }
    return strlen(str) > 0;
}

static bool mpsServerRunning(int *serverPID) {
    const int lineSize = 256;
    char line[lineSize];
    int ret;
    bool status = false;
    bool serverExist = false;

    FILE *output = popen("echo get_server_list | nvidia-cuda-mps-control 2> /dev/null", "r");
    if (!output) {
        INFO(NVSHMEM_INIT, "popen retuned NULL");
        return false;
    }

    while (fgets(line, lineSize, output) != NULL) {
        serverExist = true;
    }

    ret = pclose(output);
    status = (ret != -1) && WIFEXITED(ret) && (WEXITSTATUS(ret) == 0);
    if (!status) {
        INFO(NVSHMEM_INIT, "pclose retuned error");
        return false;
    }

    if (!serverExist) {
        return false;
    }

    if (!trimNewline(line)) {
        return false;
    }

    if (serverPID) {
        stringstream ss(line);
        int result;
        ss >> result;
        *serverPID = result;
    }

    return true;
}

static bool get_mps_server_active_thread_percentage(float *percentage) {
    FILE *output;
    const int lineSize = 256;
    char line[lineSize];
    int ret;
    char *retstr = NULL;
    bool status = false;
    stringstream cmd;
    int serverPID;
    /* one PE per node queries the control daemon */
    if (nvshmemi_state->mype == nvshmemi_team_node.start) {
        if (!mpsServerRunning(&serverPID)) {
            return false;
        }

        cmd << "echo get_active_thread_percentage " << serverPID << " | nvidia-cuda-mps-control";
        output = popen(cmd.str().c_str(), "r");

        if (!output) {
            return false;
        }

        retstr = fgets(line, lineSize, output);

        ret = pclose(output);
        status = (ret != -1) && WIFEXITED(ret) && (WEXITSTATUS(ret) == 0);
        if (!status || retstr == NULL) {
            return false;
        }

        if (!trimNewline(line)) {
            return false;
        }

        if (percentage) {
            int result;
            stringstream ss(line);
            ss >> result;
            *percentage = result;
        }
    }
    float *scratch = (float *)malloc(sizeof(float) * nvshmemi_state->npes);
    /* for lack of a better available bootstrap collective, using allagther */
    status = nvshmemi_boot_handle.allgather((void *)percentage, (void *)scratch, sizeof(float),
                                            &nvshmemi_boot_handle);
    *percentage = scratch[nvshmemi_team_node.start];
    free(scratch);

    return true;
}

static int nvshmemi_determine_mpg_support_level() {
    int status = 0;
    bool is_mps_server_running = false;
    if (nvshmemi_state->mype == nvshmemi_team_node.start) {
        is_mps_server_running = mpsServerRunning(NULL);
    }
    bool *scratch = (bool *)malloc(sizeof(bool) * nvshmemi_state->npes);
    /* for lack of a better available bootstrap collective, using allagther */
    status = nvshmemi_boot_handle.allgather((void *)&is_mps_server_running, (void *)scratch,
                                            sizeof(bool), &nvshmemi_boot_handle);
    is_mps_server_running = scratch[nvshmemi_team_node.start];
    free(scratch);

    if (!is_mps_server_running) {
        INFO(NVSHMEM_INIT,
             "Multiple PEs per GPU (MPG) detected but MPS is not running. "
             "Hence limited MPG support is available");
        nvshmemi_is_limited_mpg_run = 1;
    } else {
        float active_thread_percentage = 0;
        bool success = get_mps_server_active_thread_percentage(&active_thread_percentage);
        if (!success) {
            INFO(NVSHMEM_INIT, "failed in get_mps_server_active_thread_percentage");
            exit(-1);
        }
        char *env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
        if (env) active_thread_percentage = atof(env);

        float *active_percentages = (float *)malloc(sizeof(float) * nvshmemi_state->npes);
        status = nvshmemi_boot_handle.allgather((void *)&active_thread_percentage,
                                                (void *)active_percentages, sizeof(float),
                                                &nvshmemi_boot_handle);
        float total_percentage = 0;
        for (int i = 0; i < nvshmemi_team_same_gpu.size; i += 1) {
            total_percentage += *((float *)active_percentages + nvshmemi_team_same_gpu.start +
                                  i * nvshmemi_team_same_gpu.stride);
        }
        if (total_percentage <= 100.0) {
            nvshmemi_is_limited_mpg_run = 0;
            INFO(NVSHMEM_INIT,
                 "Multiple PEs per GPU (MPG) detected, MPS is also available, "
                 "and active thread percentages for PEs on the same GPU add "
                 "up to be <= 100. Hence full MPG support is available");
        } else {
            nvshmemi_is_limited_mpg_run = 1;
            INFO(NVSHMEM_INIT,
                 "Multiple PEs per PU (MPG) detected, MPS is also available, "
                 "but active thread percentages for PEs on the same GPU add "
                 "up to be greater than 100. Hence limited MPG support is available");
        }
        free(active_percentages);
    }
    return status;
}

static int nvshmemi_setup_limited_mpg_support() {
    int status = 0;
    nvshmemi_mps_shmdata *shm = NULL;
    nvshmemi_shared_memory_info_t *info = &nvshmemi_state->shm_info;
    cudaEvent_t event;
    int counter = 0;

    /* Ensure supported MPS runs */
    /* Do reduction to check to that each GPU has same stride and size for team_same_gpu */
    int ret = snprintf(shm_name, 100, "mps_shm_%d", nvshmemi_team_same_gpu.start);
    if (ret < 0) {
        INFO(NVSHMEM_INIT, "snprintf failed");
        return ret;
    }

    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        if (shared_memory_create(shm_name, sizeof(nvshmemi_mps_shmdata), info) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to create shared memory slab\n");
        }
    }
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    if (nvshmemi_team_same_gpu.start != nvshmemi_state->mype) {
        if (shared_memory_open(shm_name, sizeof(nvshmemi_mps_shmdata), info) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to open shared memory slab\n");
        }
    }

    shm = (nvshmemi_mps_shmdata *)info->addr;
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        shm->nprocesses = nvshmemi_team_same_gpu.size;
        shm->barrier = 0;
        shm->sense = 0;
    }
    CUDA_RUNTIME_CHECK(cudaEventCreate(&nvshmemi_state->mps_event,
                                       cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_RUNTIME_CHECK(cudaIpcGetEventHandle(
        (cudaIpcEventHandle_t *)&shm->event_handle[nvshmemi_team_same_gpu.my_pe],
        nvshmemi_state->mps_event));

    std::atomic_thread_fence(std::memory_order_seq_cst);  // flush the data
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap barrier failed \n");

    for (int i = 0; i < nvshmemi_team_same_gpu.size; i++) {
        if (i == nvshmemi_team_same_gpu.my_pe) continue;
        CUDA_RUNTIME_CHECK(
            cudaIpcOpenEventHandle(&event, *(cudaIpcEventHandle_t *)&shm->event_handle[i]));
        nvshmemi_state->same_gpu_other_pe_mps_events[counter++] = event;
    }

    /*Close the shared memory file */
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        if (info->shm_fd) {
            close(info->shm_fd);
        }
    }

out:
    return status;
}

static int nvshmemi_mpg_finalize() {
    shared_memory_close(&nvshmemi_state->shm_info);
    CUDA_RUNTIME_CHECK(cudaEventDestroy(nvshmemi_state->mps_event));
    nvshmemi_is_mpg_run = false;
    return 0;
}

int nvshmemi_common_init(nvshmemi_state_t *state) {
    int status = 0;
    cpu_set_t my_set;
    CPU_ZERO(&my_set);

    if (nvshmemi_is_nvshmem_initialized) return 0;

    if (!nvshmemi_cuda_syms) {
        nvshmemi_cuda_syms =
            (struct nvshmemi_cuda_fn_table *)calloc(1, sizeof(struct nvshmemi_cuda_fn_table));
        NVSHMEMI_NULL_ERROR_JMP(nvshmemi_cuda_syms, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate cuda function table.\n");
    }

    status = nvshmemi_cuda_library_init(nvshmemi_cuda_syms);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem cuda library init failed \n");

    CUDA_RUNTIME_CHECK(cudaDriverGetVersion(&nvshmemi_cuda_driver_version));
#if CUDA_VERSION >= 11000
    if (nvshmemi_cuda_driver_version >= 11030 && nvshmemi_options.DISABLE_CUDA_VMM == 0)
        nvshmemi_use_cuda_vmm = 1;
    else
        nvshmemi_use_cuda_vmm = 0;
#endif
    nvshmemi_state->scratch = (int *)calloc(
        nvshmemi_state->npes, sizeof(int)); /*XXX:scratch used by nvshmemi_try_common_init*/

    status = nvshmemi_get_cucontext(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem get cucontext failed \n");

    status = nvshmemi_detect_same_device(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect same device failed \n");

    status = nvshmemi_setup_stream_priorities(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem setup stream priorities failed \n");

    status = nvshmemi_setup_local_heap(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem setup local heap failed \n");

    status = nvshmemi_transport_init(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect topo failed \n");

    status = nvshmemi_build_transport_map(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "building transport map failed \n");

    status = nvshmemi_setup_cuda_handles(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuda handles setup failed \n");

    status = nvshmemi_setup_nvshmem_handles(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem handles setup failed \n");

    status = nvshmemi_setup_connections(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem setup connections failed \n");

    status = nvshmemi_setup_symmetric_heap(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup heap failed \n");

    status = nvshmemi_setup_collective_launch(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem setup collective launch failed \n");

    status = nvshmemi_init_device_state(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem device state setup failed \n");

    nvshmemi_is_nvshmem_initialized = 1;

    // coll init uses nvshmem_malloc directly
    // better to have state->initialized = 1
    status = nvshmemi_coll_common_cpu_init();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cpu collective setup failed \n");

    status = nvshmemi_coll_common_gpu_init();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gpu collective setup failed \n");

    if (sched_getaffinity(0, sizeof(my_set), &my_set) == 0) {
        int core_count = 0;

        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &my_set)) core_count++;
        }

        if (core_count == 1) {
            WARN("Proxy thread shares a core with the main PE, performance may be impacted");
        }
    }

    status = nvshmemi_proxy_init(state, nvshmemi_proxy_level(state));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy initialization failed \n");

    nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    status = nvshmemi_team_init();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "team setup failed \n");

    nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    if (nvshmemi_is_mpg_run) {
        status = nvshmemi_determine_mpg_support_level();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "call to nvshmemi_determine_mpg_support_level failed \n");
    }

    if (nvshmemi_is_limited_mpg_run) {
        status = nvshmemi_setup_limited_mpg_support();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mps setup failed \n");
    }
    nvshmemi_set_device_state(&nvshmemi_device_state);
    nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

    nvshmemi_barrier_all();
out:
    return status;
}

int nvshmemi_try_common_init(nvshmemi_state_t *state) {
    int status = 0;
    status = nvshmemi_common_init(state);
    if (status) {
        INFO(NVSHMEM_INIT, "nvshmemi_common_init failed, continuing");
        status = 0;
    }

    return status;
}

int nvshmemx_internal_common_init() {
    int status = 0;
    status = nvshmemi_common_init(nvshmemi_state);
    if (status) {
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemi_common_init failed ...");
    }
out:
    return status;
}

void nvshmemx_get_device_state(nvshmemi_device_state_t **device_state) {
    *device_state = &nvshmemi_device_state;
}

int nvshmemx_internal_init_thread(int requested, int *provided, unsigned int bootstrap_flags,
                                  nvshmemx_init_attr_t *attr,
                                  nvshmemi_version_t nvshmem_device_lib_version) {
    if (nvshmemi_is_version_compatible(nvshmemi_host_lib_version, nvshmem_device_lib_version) !=
        0) {
        printf("NVSHMEM device library version does not match with NVSHMEM host library version\n");
        return 1;
    }

    int status = 0;

    if (!nvshmemi_is_nvshmem_bootstrapped) {
        nvshmemi_options_init();
        nvshmem_nvtx_init();
    }

    NVTX_FUNC_RANGE_IN_GROUP(INIT);

    if (!nvshmemi_is_nvshmem_bootstrapped) {
        NVSHMEMU_THREAD_CS_INIT();
        nvshmemi_init_debug();

        status = nvshmemi_bootstrap(bootstrap_flags, attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_bootstrap failed \n");

        nvshmemi_init_msg();

        nvshmemi_is_nvshmem_bootstrapped = true;
        atexit(bootstrap_finalize);
    }

    if (!nvshmemi_is_nvshmem_initialized) {
        nvshmemi_state = (nvshmemi_state_t *)calloc(1, sizeof(nvshmemi_state_t));
        NVSHMEMI_NULL_ERROR_JMP(nvshmemi_state, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "nvshmemi_init_thread/calloc failed \n");
        nvshmemi_init_nvshmemi_state(nvshmemi_state);

        status = nvshmemi_try_common_init(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem common init failed \n");
    }

    *provided = NVSHMEM_THREAD_SERIALIZED;

out:
    if (status) NVSHMEMU_THREAD_CS_FINALIZE();

    return status;
}

void nvshmem_query_thread(int *provided) { *provided = NVSHMEM_THREAD_SERIALIZED; }

void nvshmemx_query_thread(int *provided) { nvshmem_query_thread(provided); }

#ifndef __CUDA_ARCH__
void nvshmem_global_exit(int status) {
    nvshmemi_is_nvshmem_bootstrapped =
        false; /* Set it to 0 so that atexit does not try to finalize_bootstrap */
    /* We can't fix anything if the call to nvshmemi_proxy_finalize fails so don't check the error
     * message. We need to stop the proxy thread before calling global exit to stop a race between
     * the proxy and the atexit bootstrap_finalize function.
     */
    nvshmemi_proxy_finalize(nvshmemi_state);
    nvshmemi_boot_handle.global_exit(status);
}
#endif

void nvshmemi_finalize() {
    NVTX_FUNC_RANGE_IN_GROUP(INIT);
    nvshmemi_init_counter--;
    if (nvshmemi_init_counter != 0) return;

    int status = 0;
    int pid = getpid();
    INFO(NVSHMEM_INIT, "[%d] in nvshmem_finalize:", pid);

    if (nvshmemi_is_nvshmem_initialized) {
        nvshmemi_barrier_all();
        nvshmemx_quiet_on_stream(
            nvshmemi_state->my_stream); /* wait for signal ops from barrier to complete */
        status = cudaDeviceSynchronize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Teams cleanup device synchronization failed \n");

        /* barrier to ensure all previous ops are complete */
        nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

        /* mps finalize */
        if (nvshmemi_is_limited_mpg_run) {
            status = nvshmemi_mpg_finalize();
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "MPS cleanup failed \n");
        }

        /* teams cleanup */
        status = nvshmemi_team_finalize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Teams cleanup failed \n");

        /*cleaning up proxy*/
        if (nvshmemi_proxy_level(nvshmemi_state)) {
            status = nvshmemi_proxy_finalize(nvshmemi_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy cleanup failed \n");
        }

        /* collective cleanup */
        status = nvshmemi_coll_common_cpu_finalize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "CPU collectives cleanup failed \n");

        status = nvshmemi_coll_common_gpu_finalize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "GPU collectives cleanup failed \n");

        status = nvshmemi_teardown_handles(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "handles cleanup failed \n");

        status = nvshmemi_cleanup_symmetric_heap(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "symmetric heap cleanup failed \n");

        status = nvshmemi_transport_finalize(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem transport finalize failed \n");

        status = nvshmemi_teardown_collective_launch(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "collective launch cleanup failed \n");

        /* cleanup state */
        if (nvshmemi_state->scratch_size) free(nvshmemi_state->scratch_space);
        if (nvshmemi_state->scratch) free(nvshmemi_state->scratch);
        free(nvshmemi_state);

        nvshmemi_is_nvshmem_initialized = 0;
        nvshmemi_is_device_state_set = 0;

    } else
        nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("aborting due to error in nvshmem_finalize \n");
        exit(-1);
    }
}

static void nvshmemi_init_debug() {
    const char *nvshmem_debug = nvshmemi_options.DEBUG;
    if (!nvshmemi_options.DEBUG_provided && !nvshmemi_options.DEBUG_SUBSYS_provided) {
        nvshmem_debug_level = NVSHMEM_LOG_NONE;
    } else if (strcmp_case_insensitive(nvshmem_debug, "VERSION") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_VERSION;
    } else if (strcmp_case_insensitive(nvshmem_debug, "WARN") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_WARN;
    } else if (strcmp_case_insensitive(nvshmem_debug, "INFO") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    } else if (strcmp_case_insensitive(nvshmem_debug, "ABORT") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_ABORT;
    } else if (strcmp_case_insensitive(nvshmem_debug, "TRACE") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_TRACE;
    } else {
        /* OpenSHMEM spec treats SHMEM_DEBUG as a boolean, enable INFO logging
         * when user-supplied value does match one of the above. */
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    }

    /* Parse the NVSHMEM_DEBUG_SUBSYS env var
     * This can be a comma separated list such as INIT,COLL
     * or ^INIT,COLL etc
     */
    /* Note: strtok will modify the string, operate on a copy */
    char *nvshmem_debug_subsys = strdup(nvshmemi_options.DEBUG_SUBSYS);
    if (nvshmem_debug_subsys != NULL) {
        char *subsys = strtok(nvshmem_debug_subsys, ",");
        while (subsys != NULL) {
            int invert = 0;
            uint64_t mask = 0;
            if (subsys[0] == '^') {
                invert = 1;
                subsys++;
            }
            if (strcmp_case_insensitive(subsys, "INIT") == 0) {
                mask = NVSHMEM_INIT;
            } else if (strcmp_case_insensitive(subsys, "COLL") == 0) {
                mask = NVSHMEM_COLL;
            } else if (strcmp_case_insensitive(subsys, "P2P") == 0) {
                mask = NVSHMEM_P2P;
            } else if (strcmp_case_insensitive(subsys, "PROXY") == 0) {
                mask = NVSHMEM_PROXY;
            } else if (strcmp_case_insensitive(subsys, "TRANSPORT") == 0) {
                mask = NVSHMEM_TRANSPORT;
            } else if (strcmp_case_insensitive(subsys, "MEM") == 0) {
                mask = NVSHMEM_MEM;
            } else if (strcmp_case_insensitive(subsys, "BOOTSTRAP") == 0) {
                mask = NVSHMEM_BOOTSTRAP;
            } else if (strcmp_case_insensitive(subsys, "TOPO") == 0) {
                mask = NVSHMEM_TOPO;
            } else if (strcmp_case_insensitive(subsys, "UTIL") == 0) {
                mask = NVSHMEM_UTIL;
            } else if (strcmp_case_insensitive(subsys, "ALL") == 0) {
                mask = NVSHMEM_ALL;
            } else {
                mask = 0;
                WARN("Unrecognized value in DEBUG_SUBSYS: %s%s", invert ? "^" : "", subsys);
            }
            if (mask) {
                if (invert)
                    nvshmem_debug_mask &= ~mask;
                else
                    nvshmem_debug_mask |= mask;
            }
            subsys = strtok(NULL, ",");
        }

        free(nvshmem_debug_subsys);
    }

    /* Parse and expand the NVSHMEM_DEBUG_FILE path and
     * then create the debug file. But don't bother unless the
     * NVSHMEM_DEBUG level is > VERSION
     */
    const char *nvshmem_debug_filename = nvshmemi_options.DEBUG_FILE;
    if (nvshmem_debug_level > NVSHMEM_LOG_VERSION && nvshmemi_options.DEBUG_FILE_provided) {
        int c = 0;
        char debugFn[PATH_MAX + 1] = "";
        char *dfn = debugFn;
        while (nvshmem_debug_filename[c] != '\0' && c < PATH_MAX) {
            if (nvshmem_debug_filename[c++] != '%') {
                *dfn++ = nvshmem_debug_filename[c - 1];
                continue;
            }
            switch (nvshmem_debug_filename[c++]) {
                case '%':  // Double %
                    *dfn++ = '%';
                    break;
                case 'h':  // %h = hostname
                    char hostname[1024];
                    nvshmemu_gethostname(hostname, 1024);
                    dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
                    break;
                case 'p':  // %p = pid
                    dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
                    break;
                default:  // Echo everything we don't understand
                    *dfn++ = '%';
                    *dfn++ = nvshmem_debug_filename[c - 1];
                    break;
            }
        }
        *dfn = '\0';
        if (debugFn[0] != '\0') {
            FILE *file = fopen(debugFn, "w");
            if (file != NULL) {
                INFO(NVSHMEM_ALL, "DEBUG file is '%s'", debugFn);
                nvshmem_debug_file = file;
            }
        }
    }
    pthread_mutex_init(&nvshmem_debug_output_lock, NULL);

#ifdef NVSHMEM_TRACE
    nvshmem_epoch = std::chrono::high_resolution_clock::now();
#endif
}

static void nvshmemi_init_msg(void) {
    if (0 == nvshmemi_boot_handle.pg_rank) {
        if (nvshmemi_options.VERSION) printf("%s\n", NVSHMEM_VENDOR_STRING);

        if (nvshmemi_options.DEBUG_provided) {
            int runtimeVersion, driverVersion;
            cudaError_t err;

            printf("NVSHMEM configuration:\n");

            printf("  %-28s %d\n", "CUDA API", CUDA_VERSION);

            err = cudaRuntimeGetVersion(&runtimeVersion);
            if (err != cudaSuccess) runtimeVersion = -1;
            printf("  %-28s %d\n", "CUDA Runtime", runtimeVersion);

            err = cudaDriverGetVersion(&driverVersion);
            if (err != cudaSuccess) driverVersion = -1;
            printf("  %-28s %d\n", "CUDA Driver", driverVersion);

            printf("  %-28s %s %s\n", "Build Timestamp", __DATE__, __TIME__);

            char *build_vars = nvshmemu_wrap(NVSHMEM_BUILD_VARS, NVSHMEMI_WRAPLEN, "\t", 0);
            printf("  %-28s\n\t%s\n", "Build Variables",
                   build_vars ? build_vars : "Error wrapping build vars");
            free(build_vars);

            printf("\n");
        }
        if (nvshmemi_options.INFO) nvshmemi_options_print(NVSHMEMI_OPTIONS_STYLE_INFO);
    }

    if (nvshmemi_options.DEBUG_provided || nvshmemi_options.DEBUG_SUBSYS_provided)
        nvshmemu_debug_log_cpuset(NVSHMEM_INIT, "process");
}

int nvshmemi_proxy_level(nvshmemi_state_t *state) {
    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (state->transports[i]->is_successfully_initialized) {
            if (state->transports[i]->no_proxy) {
                continue;
            } else {
                return NVSHMEMI_PROXY_FULL;
            }
        }
    }

    if (nvshmemi_options.DISABLE_LOCAL_ONLY_PROXY) {
        return NVSHMEMI_PROXY_NONE;
    }

    return NVSHMEMI_PROXY_MINIMAL;
}

int set_job_connectivity(nvshmemi_state_t *state) {
    int status;
    int *job_connectivity_all;
    bool proxy_ops_are_ordered = true;
    int gpu_remote_atomics = false;

    // determine job level connectivity among GPUs
    nvshmemi_job_connectivity = NVSHMEMI_JOB_GPU_LDST_ATOMICS;
    for (int i = 0; i < state->npes; i++) {
        int peer_connectivity = NVSHMEMI_JOB_GPU_PROXY;
        void *enforce_cst = NULL;
        // for each PE, pick the best connectivity of any transport
        for (int j = 0; j < state->num_initialized_transports; j++) {
            if (state->transports[j]) {
                if (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS) {
                    peer_connectivity = (int)NVSHMEMI_JOB_GPU_LDST_ATOMICS;
                } else if (state->transports[j]->cap[i] &
                           (NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD)) {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_LDST);
                }
#ifdef NVSHMEM_IBGDA_SUPPORT
                else if (state->transports[j]->cap[i] &
                         (NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
                          NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS)) {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_PROXY);
                    /* Note, these are not mapped atomics. They would be atomics issued from the GPU
                     * over a remote transport (e.g. GIC). */
                    if (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS) {
                        gpu_remote_atomics = true;
                    }
                }
#endif
                else {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_PROXY);
                    enforce_cst = (void *)state->transports[j]->host_ops.enforce_cst_at_target;
                }
            }
        }

        if ((peer_connectivity == NVSHMEMI_JOB_GPU_PROXY) && (enforce_cst)) {
            peer_connectivity = NVSHMEMI_JOB_GPU_PROXY_CST;
        }

        // for the job, pick the weakest connecitivity to any remote PEs
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, peer_connectivity);
    }

    /* This case allows us to differentiate between cases where we only support LDST
     * and cases where we have LDST + atomics over a remote transport elsewhere in the code.
     * This catches cases where the remote transport either does, or does not have a proxy.
     */
    gpu_remote_atomics =
        nvshmemi_proxy_level(state) == NVSHMEMI_PROXY_FULL ? true : gpu_remote_atomics;
    if (nvshmemi_job_connectivity == NVSHMEMI_JOB_GPU_LDST && gpu_remote_atomics) {
        nvshmemi_job_connectivity = NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS;
    }

    // agree on maximumg distance for job_connectivity among all PEs
    job_connectivity_all = (int *)malloc(sizeof(int) * state->npes);
    NVSHMEMI_NULL_ERROR_JMP(job_connectivity_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "memory allocation for job_connectivity_all failed \n");

    status = nvshmemi_boot_handle.allgather((void *)&nvshmemi_job_connectivity,
                                            (void *)job_connectivity_all, sizeof(int),
                                            &nvshmemi_boot_handle);
    if (status != 0) {
        free(job_connectivity_all);
        NVSHMEMI_ERROR_PRINT("allgather of job_connectivity failed \n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    for (int i = 0; i < state->npes; i++) {
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, job_connectivity_all[i]);
    }
    free(job_connectivity_all);
    nvshmemi_device_state.job_connectivity = nvshmemi_job_connectivity;

    // check if all proxy ops are ordered
    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (state->transports[i] && (state->transports[i]->host_ops.fence != NULL))
            proxy_ops_are_ordered = false;
    }
    nvshmemi_device_state.proxy_ops_are_ordered = proxy_ops_are_ordered;

out:
    return status;
}

int nvshmemi_init_device_state(nvshmemi_state_t *state) {
    int status = CUDA_SUCCESS;
    int warp_size = 0;

    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, state->device_id), status, out);
    if (NVSHMEMI_WARP_SIZE != warp_size) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "device warp size (%d) does not match assumed warp size (%d)\n",
                              warp_size, NVSHMEMI_WARP_SIZE);
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaMalloc(&heap_base_array_dptr, (state->npes) * sizeof(void *)),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaMalloc(&heap_base_actual_array_dptr, (state->npes) * sizeof(void *)), status, out);

    status = set_job_connectivity(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "set_job_connectivity failed \n");

    CUDA_RUNTIME_CHECK_GOTO(
        cudaMemcpyAsync(heap_base_array_dptr, (const void *)state->peer_heap_base,
                        sizeof(void *) * state->npes, cudaMemcpyHostToDevice, state->my_stream),
        status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaMemcpyAsync(heap_base_actual_array_dptr, (const void *)state->peer_heap_base_actual,
                        sizeof(void *) * state->npes, cudaMemcpyHostToDevice, state->my_stream),
        status, out);

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(state->my_stream), status, out);

    nvshmemi_device_state.proxy = nvshmemi_proxy_level(state);

    if (nvshmemi_options.ASSERT_ATOMICS_SYNC)
        nvshmemi_device_state.atomics_sync = 1;
    else
        nvshmemi_device_state.atomics_sync = 0;

    nvshmemi_device_state.atomics_le_min_size = state->atomic_host_endian_min_size;

    for (int i = 0; i < state->npes; i++) {
        int t_idx = state->selected_transport_for_amo[i];
        if (t_idx < 0 || t_idx >= NVSHMEM_TRANSPORT_COUNT) {
            continue;
        }
        if (state->transports[t_idx]->atomics_complete_on_quiet) {
            nvshmemi_device_state.atomics_complete_on_quiet = true;
            break;
        }
    }

    nvshmemi_device_state.peer_heap_base = (void **)heap_base_array_dptr;

    INFO(NVSHMEM_INIT,
         "[%d] status %d cudaErrorInvalidValue %d cudaErrorInvalidSymbol %d "
         "cudaErrorInvalidMemcpyDirection %d cudaErrorNoKernelImageForDevice %d",
         state->mype, status, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
         cudaErrorInvalidMemcpyDirection, cudaErrorNoKernelImageForDevice);

    nvshmemi_device_state.peer_heap_base_actual = (void **)heap_base_actual_array_dptr;
    nvshmemi_device_state.heap_base = state->heap_base;
    nvshmemi_device_state.heap_size = state->heap_size;
    nvshmemi_device_state.mype = state->mype;
    nvshmemi_device_state.npes = state->npes;
    nvshmemi_device_state.node_mype = state->mype_node;
    nvshmemi_device_state.node_npes = state->npes_node;
    nvshmemi_device_state.barrier_dissem_kval = nvshmemi_options.BARRIER_DISSEM_KVAL;
    nvshmemi_device_state.barrier_tg_dissem_kval = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(state->my_stream), status, out);

    unsigned long long *test_wait_any_start_idx_ptr;
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&test_wait_any_start_idx_ptr, sizeof(unsigned long long)));
    CUDA_RUNTIME_CHECK(
        cudaMemset((void *)test_wait_any_start_idx_ptr, 0, sizeof(unsigned long long)));

    nvshmemi_device_state.test_wait_any_start_idx_ptr = test_wait_any_start_idx_ptr;

    nvshmemi_set_device_state(&nvshmemi_device_state);

out:
    if (status) {
        if (heap_base_array_dptr) free(heap_base_array_dptr);
        if (heap_base_actual_array_dptr) free(heap_base_actual_array_dptr);
        if (test_wait_any_start_idx_ptr) cudaFree(test_wait_any_start_idx_ptr);
    }
    return status;
}

static void register_ibgda_state_ptr(void *arg) { registered_ibgda_states.emplace(arg); }

int nvshmemx_cumodule_init(CUmodule module) {
    int status = 0;
    CUdeviceptr dptr;
    size_t size;
    nvshmemi_version_t module_nvshmem_version;

    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_device_lib_version_d"), status,
                out);
    CUDA_RUNTIME_CHECK(cudaMemcpy((void *)&module_nvshmem_version, (const void *)dptr, size,
                                  cudaMemcpyDeviceToHost));
    if (nvshmemi_is_version_compatible(module_nvshmem_version, nvshmemi_host_lib_version) != 0) {
        printf("NVSHMEM version in CUmodule does not match with NVSHMEM host library version\n");
        return 1;
    }

    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_device_state_d"), status, out);
    status = cudaMemcpyFromSymbol((void *)dptr, nvshmemi_device_state_d, size, 0,
                                  cudaMemcpyDeviceToDevice);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemcpyFromSymbol failed\n");
    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_gic_device_state_d"), status,
                out);
    register_ibgda_state_ptr((void *)dptr);
    status = cudaDeviceSynchronize();
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemcpyFromSymbol failed\n");
out:
    return status;
}

void nvshmemi_register_state_change_handler(nvshmemi_state_change_handler_fn_t fn) {
    state_change_handler_set.emplace(fn);
}

#ifdef NVSHMEM_IBGDA_SUPPORT
static int update_registered_ibgda_device_states() {
    nvshmemi_gic_device_state_t *gic_device_state;
    int status = 0;

    nvshmemx_gic_get_device_state((void **)&gic_device_state);
    for (auto it = registered_ibgda_states.cbegin(); it != registered_ibgda_states.cend(); ++it) {
        status = cudaMemcpy(*it, (void *)gic_device_state, sizeof(nvshmemi_gic_device_state_t),
                            cudaMemcpyHostToDevice);
        if (status) break;
    }

    return status;
}
#endif

int nvshmemi_update_device_state() {
    int status = 0;

#ifdef NVSHMEM_IBGDA_SUPPORT
    status = update_registered_ibgda_device_states();
    if (status) {
        goto out;
    }
#endif

    for (auto it = state_change_handler_set.cbegin(); it != state_change_handler_set.cend(); ++it) {
        status = (*it)();
        if (status) {
            goto out;
        }
    }

out:

    return status;
}
