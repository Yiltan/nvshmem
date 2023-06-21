/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

/* NVSHMEMI_ENV_DEF( name, kind, default, category, short description )
 *
 * Kinds: long, size, bool, string
 * Categories: NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
 *             NVSHMEMI_ENV_CAT_COLLECTIVES, NVSHMEMI_ENV_CAT_TRANSPORT,
 *             NVSHMEMI_ENV_CAT_HIDDEN
 */

/*
 * Preincluded header requirements: nvshmem_internal.h
 */

/* Header part one, definitions and functions. Only include once. */
#include <math.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

#ifndef NVSHMEM_ENV_DEFS
#define NVSHMEM_ENV_DEFS

#define SYMMETRIC_SIZE_DEFAULT 1024 * 1024 * 1024

typedef int nvshmemi_env_int;
typedef long nvshmemi_env_long;
typedef size_t nvshmemi_env_size;
typedef bool nvshmemi_env_bool;
typedef const char *nvshmemi_env_string;

#define NVSHFMT_int(_v) _v
#define NVSHFMT_long(_v) _v
#define NVSHFMT_size(_v) _v
#define NVSHFMT_bool(_v) (_v) ? "true" : "false"
#define NVSHFMT_string(_v) _v

enum nvshmemi_env_categories {
    NVSHMEMI_ENV_CAT_OPENSHMEM,
    NVSHMEMI_ENV_CAT_OTHER,
    NVSHMEMI_ENV_CAT_COLLECTIVES,
    NVSHMEMI_ENV_CAT_TRANSPORT,
    NVSHMEMI_ENV_CAT_HIDDEN,
    NVSHMEMI_ENV_CAT_NVTX,
    NVSHMEMI_ENV_CAT_BOOTSTRAP
};

struct nvshmemi_options_s {
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) nvshmemi_env_##KIND NAME;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) bool NAME##_provided;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
};

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static inline int nvshmemi_atol_scaled(const char *str, nvshmemi_env_size *out) {
    int scale, n;
    double p = -1.0;
    char f;

    n = sscanf(str, "%lf%c", &p, &f);

    if (n == 2) {
        switch (f) {
            case 'k':
            case 'K':
                scale = 10;
                break;
            case 'm':
            case 'M':
                scale = 20;
                break;
            case 'g':
            case 'G':
                scale = 30;
                break;
            case 't':
            case 'T':
                scale = 40;
                break;
            default:
                return 1;
        }
    } else if (p < 0) {
        return 1;
    } else
        scale = 0;

    *out = (nvshmemi_env_size)ceil(p * (1lu << scale));
    return 0;
}

static inline long nvshmemi_errchk_atol(const char *s) {
    long val;
    char *e;
    errno = 0;

    val = strtol(s, &e, 0);
    if (errno != 0 || e == s) {
        fprintf(stderr, "Environment variable conversion failed (%s)\n", s);
    }

    return val;
}

static inline const char *nvshmemi_getenv_helper(const char *prefix, const char *name) {
    char *env_name;
    const char *env_value = NULL;
    size_t len;
    int ret;

    len = strlen(prefix) + 1 /* '_' */ + strlen(name) + 1 /* '\0' */;
    env_name = (char *)alloca(len);
    ret = snprintf(env_name, len, "%s_%s", prefix, name);
    if (ret < 0)
        fprintf(stderr, "WARNING: Error in sprintf: %s_%s\n", prefix, name);
    else
        env_value = (const char *)getenv(env_name);

    return env_value;
}

static inline const char *nvshmemi_getenv(const char *name) {
    const char *env_value;

    env_value = nvshmemi_getenv_helper("NVSHMEM", name);
    if (env_value != NULL) return env_value;

    return NULL;
}

static inline int nvshmemi_getenv_string(const char *name, nvshmemi_env_string default_val,
                                         nvshmemi_env_string *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? env : default_val;
    return 0;
}

static inline int nvshmemi_getenv_int(const char *name, nvshmemi_env_int default_val,
                                      nvshmemi_env_int *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? (int)nvshmemi_errchk_atol(env) : default_val;
    return 0;
}

static inline int nvshmemi_getenv_long(const char *name, nvshmemi_env_long default_val,
                                       nvshmemi_env_long *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? nvshmemi_errchk_atol(env) : default_val;
    return 0;
}

static inline int nvshmemi_getenv_size(const char *name, nvshmemi_env_size default_val,
                                       nvshmemi_env_size *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    if (*provided) {
        int ret = nvshmemi_atol_scaled(env, out);
        if (ret) {
            fprintf(stderr, "Invalid size in environment variable '%s' (%s)\n", name, env);
            return ret;
        }
    } else
        *out = default_val;
    return 0;
}

static inline int nvshmemi_getenv_bool(const char *name, nvshmemi_env_bool default_val,
                                       nvshmemi_env_bool *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);

    if (*provided &&
        (env[0] == '0' || env[0] == 'N' || env[0] == 'n' || env[0] == 'F' || env[0] == 'f')) {
        *out = false;
    } else if (*provided) {
        /* The default behavior specified by OpenSHMEM is to enable boolean
         * options whenever the environment variable is set */
        *out = true;
    } else {
        *out = default_val;
    }

    return 0;
}

static inline int nvshmemi_env_options_init(struct nvshmemi_options_s *options) {
    int ret;
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                              \
    ret = nvshmemi_getenv_##KIND(#NAME, DEFAULT, &(options->NAME), &(options->NAME##_provided)); \
    if (ret) return ret;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    return 0;
}
#endif

/* Header part two, environment variable names. Can be included as much as possible.
 * Note, this portion relies on the including file defining NVSHMEMI_ENV_DEF.
 */
#ifdef NVSHMEMI_ENV_DEF

NVSHMEMI_ENV_DEF(VERSION, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print library version at startup")
NVSHMEMI_ENV_DEF(INFO, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print environment variable options at startup")
NVSHMEMI_ENV_DEF(INFO_HIDDEN, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Print hidden environment variable options at startup")
NVSHMEMI_ENV_DEF(SYMMETRIC_SIZE, size, (size_t)(SYMMETRIC_SIZE_DEFAULT), NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Specifies the size (in bytes) of the symmetric heap memory per PE. The resulting "
                 "size is implementation-defined and must be at least as large as the integer "
                 "ceiling of the product of the numeric prefix and the scaling factor. The allowed "
                 "character suffixes for the scaling factor are as follows:\n"
                 "\n"
                 "  *  k or K multiplies by 2^10 (kibibytes)\n"
                 "  *  m or M multiplies by 2^20 (mebibytes)\n"
                 "  *  g or G multiplies by 2^30 (gibibytes)\n"
                 "  *  t or T multiplies by 2^40 (tebibytes)\n"
                 "\n"
                 "For example, string '20m' is equivalent to the integer value 20971520, or 20 "
                 "mebibytes. Similarly the string '3.1M' is equivalent to the integer value "
                 "3250586. Only one multiplier is recognized and any characters following the "
                 "multiplier are ignored, so '20kk' will not produce the same result as '20m'. "
                 "Usage of string '.5m' will yield the same result as the string '0.5m'.\n"
                 "An invalid value for ``NVSHMEM_SYMMETRIC_SIZE`` is an error, which the NVSHMEM "
                 "library shall report by either returning a nonzero value from "
                 "``nvshmem_init_thread`` or causing program termination.")
NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "Optional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Bootstrap **/

NVSHMEMI_ENV_DEF(BOOTSTRAP, string, "PMI", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the default bootstrap that should be used to initialize NVSHMEM.\n"
                 "Allowed values: PMI, MPI, SHMEM, plugin")

#if defined(NVSHMEM_DEFAULT_PMIX)
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMIX"
#elif defined(NVSHMEM_DEFAULT_PMI2)
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMI-2"
#else
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMI"
#endif

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI, string, NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT,
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the PMI bootstrap that should be used to initialize NVSHMEM.\n"
                 "Allowed values: PMI, PMI-2, PMIX")

#undef NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT

NVSHMEMI_ENV_DEF(BOOTSTRAP_PLUGIN, string, "", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the bootstrap plugin file to load when NVSHMEM_BOOTSTRAP=plugin "
                 "is specified")

NVSHMEMI_ENV_DEF(BOOTSTRAP_MPI_PLUGIN, string, "nvshmem_bootstrap_mpi.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP, "Name of the MPI bootstrap plugin file")

NVSHMEMI_ENV_DEF(BOOTSTRAP_SHMEM_PLUGIN, string, "nvshmem_bootstrap_shmem.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP, "Name of the SHMEM bootstrap plugin file")

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI_PLUGIN, string, "nvshmem_bootstrap_pmi.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP, "Name of the PMI bootstrap plugin file")

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI2_PLUGIN, string, "nvshmem_bootstrap_pmi2.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP, "Name of the PMI-2 bootstrap plugin file")

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMIX_PLUGIN, string, "nvshmem_bootstrap_pmix.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP, "Name of the PMIx bootstrap plugin file")

NVSHMEMI_ENV_DEF(BOOTSTRAP_TWO_STAGE, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Ignore CUDA device setting during initialization,"
                 "forcing two-stage initialization")

/** Debugging **/

NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix with '^' to exclude.\n"
                 "Values: INIT, COLL, P2P, PROXY, TRANSPORT, MEM, BOOTSTRAP, TOPO, UTIL, ALL")
NVSHMEMI_ENV_DEF(DEBUG_FILE, string, "", NVSHMEMI_ENV_CAT_OTHER,
                 "Debugging output filename, may contain %h for hostname and %p for pid")
NVSHMEMI_ENV_DEF(ENABLE_ERROR_CHECKS, bool, false, NVSHMEMI_ENV_CAT_HIDDEN, "Enable error checks")

NVSHMEMI_ENV_DEF(MAX_TEAMS, long, 32l, NVSHMEMI_ENV_CAT_OTHER,
                 "Maximum number of simultaneous teams allowed")

NVSHMEMI_ENV_DEF(MAX_P2P_GPUS, int, 128, NVSHMEMI_ENV_CAT_OTHER, "Maximum number of P2P GPUs")
NVSHMEMI_ENV_DEF(MAX_MEMORY_PER_GPU, size, (size_t)((size_t)128 * (1 << 30)),
                 NVSHMEMI_ENV_CAT_OTHER, "Maximum memory per GPU")
#if defined(NVSHMEM_PPC64LE)
#define NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT true
#else
#define NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT false
#endif

NVSHMEMI_ENV_DEF(DISABLE_CUDA_VMM, bool, NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT,
                 NVSHMEMI_ENV_CAT_OTHER,
                 "Disable use of CUDA VMM for P2P memory mapping. By default, CUDA VMM is enabled "
                 "on x86 and disabled on P9. CUDA VMM feature in NVSHMEM requires CUDA RT version "
                 "and CUDA Driver version to be greater than or equal to 11.3.")

#undef NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT

NVSHMEMI_ENV_DEF(DISABLE_P2P, bool, false, NVSHMEMI_ENV_CAT_OTHER,
                 "Disable P2P connectivity of GPUs even when available")
NVSHMEMI_ENV_DEF(CUMEM_GRANULARITY, size, (size_t)((size_t)1 << 29), NVSHMEMI_ENV_CAT_OTHER,
                 "Granularity for ``cuMemAlloc``/``cuMemCreate``")

NVSHMEMI_ENV_DEF(BYPASS_ACCESSIBILITY_CHECK, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass peer GPU accessbility checks")

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_ENV_ALL)
NVSHMEMI_ENV_DEF(CUDA_LIMIT_STACK_SIZE, size, (size_t)(0), NVSHMEMI_ENV_CAT_OTHER,
                 "Specify limit on stack size of each GPU thread on P9")
#endif

/** General Collectives **/

NVSHMEMI_ENV_DEF(DISABLE_NCCL, bool, false, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Disable use of NCCL for collective operations")
NVSHMEMI_ENV_DEF(BARRIER_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for barriers")
NVSHMEMI_ENV_DEF(BARRIER_TG_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for thread group barriers")
NVSHMEMI_ENV_DEF(REDUCE_RECEXCH_KVAL, int, 2, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Radix of the recursive exchange reduction algorithm")
NVSHMEMI_ENV_DEF(FCOLLECT_LL_THRESHOLD, size, (size_t)(1 << 11), NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Message size threshold up to which "
                 "fcollect LL algo will be used")
NVSHMEMI_ENV_DEF(BCAST_TREE_KVAL, int, 2, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Radix of the broadcast tree algorithm")
NVSHMEMI_ENV_DEF(BCAST_ALGO, int, 0, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Broadcast algorithm to be used.\n"
                 "  * 0 - use default algorithm selection strategy\n")
NVSHMEMI_ENV_DEF(REDMAXLOC_ALGO, int, 1, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Reduction algorithm to be used.\n"
                 "  * 1 - default, flag alltoall algorithm\n"
                 "  * 2 - flat reduce + flat bcast\n"
                 "  * 3 - topo-aware two-level reduce + topo-aware bcast\n")

/** CPU Collectives **/

NVSHMEMI_ENV_DEF(RDX_NUM_TPB, int, 32, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of threads per block used for reduction purposes")

/** Transport **/

#ifdef NVSHMEM_DEFAULT_UCX
#define NVSHMEMI_ENV_TRANSPORT_DEFAULT "ucx"
#else
#define NVSHMEMI_ENV_TRANSPORT_DEFAULT "ibrc"
#endif

NVSHMEMI_ENV_DEF(ASSERT_ATOMICS_SYNC, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush on wait_until at target")
NVSHMEMI_ENV_DEF(REMOTE_TRANSPORT, string, NVSHMEMI_ENV_TRANSPORT_DEFAULT,
                 NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Selected transport for remote operations: ibrc, ucx, libfabric, ibdevx, none")
NVSHMEMI_ENV_DEF(DISABLE_IB_NATIVE_ATOMICS, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of InfiniBand native atomics")
NVSHMEMI_ENV_DEF(DISABLE_GDRCOPY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of GDRCopy in IB RC Transport")
NVSHMEMI_ENV_DEF(BYPASS_FLUSH, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush in proxy when enforcing consistency")
NVSHMEMI_ENV_DEF(ENABLE_NIC_PE_MAPPING, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When not set or set to 0, a PE is assigned the NIC on the node that is "
                 "closest to it by distance. When set to 1, NVSHMEM either assigns NICs to "
                 "PEs on a round-robin basis or uses ``NVSHMEM_HCA_PE_MAPPING`` or "
                 "``NVSHMEM_HCA_LIST`` when they are specified.")
NVSHMEMI_ENV_DEF(IB_GID_INDEX, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT, "Source GID Index for ROCE")
NVSHMEMI_ENV_DEF(IB_TRAFFIC_CLASS, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT, "Traffic calss for ROCE")
NVSHMEMI_ENV_DEF(IB_SL, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT, "Service level to use over IB/ROCE")

NVSHMEMI_ENV_DEF(HCA_LIST, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Comma-separated list of HCAs to use in the NVSHMEM application. Entries "
                 "are of the form ``hca_name:port``, e.g. ``mlx5_1:1,mlx5_2:2`` and entries "
                 "prefixed by ^ are excluded. ``NVSHMEM_ENABLE_NIC_PE_MAPPING`` must be set to "
                 "1 for this variable to be effective.")

NVSHMEMI_ENV_DEF(HCA_PE_MAPPING, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies mapping of HCAs to PEs as a comma-separated list. Each entry "
                 "in the comma separated list is of the form ``hca_name:port:count``.  For "
                 "example, ``mlx5_0:1:2,mlx5_0:2:2`` indicates that PE0, PE1 are mapped to "
                 "port 1 of mlx5_0, and PE2, PE3 are mapped to port 2 of mlx5_0. "
                 "``NVSHMEM_ENABLE_NIC_PE_MAPPING`` must be set to 1 for this variable to be "
                 "effective.")
NVSHMEMI_ENV_DEF(QP_DEPTH, int, 1024, NVSHMEMI_ENV_CAT_HIDDEN, "Number of WRs in QP")
NVSHMEMI_ENV_DEF(SRQ_DEPTH, int, 16384, NVSHMEMI_ENV_CAT_HIDDEN, "Number of WRs in SRQ")

NVSHMEMI_ENV_DEF(DISABLE_LOCAL_ONLY_PROXY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When running on an NVLink-only configuaration (No-IB, No-UCX), completely "
                 "disable the proxy thread. This will disable device side global exit and "
                 "device side wait timeout polling (enabled by ``NVSHMEM_TIMEOUT_DEVICE_POLLING`` "
                 "build-time variable) because these are processed by the proxy thread.")

NVSHMEMI_ENV_DEF(LIBFABRIC_PERSONA, string, "cxi", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Set the feature set persona for the libfabric transport: cxi, verbs")

/** Runtime optimimzations **/
NVSHMEMI_ENV_DEF(PROXY_REQUEST_BATCH_MAX, int, 32, NVSHMEMI_ENV_CAT_OTHER,
                 "Maxmum number of requests that the proxy thread processes in a single iteration "
                 "of the progress loop.")

/** NVTX instrumentation **/
NVSHMEMI_ENV_DEF(NVTX, string, "off", NVSHMEMI_ENV_CAT_NVTX,
                 "Set to enable NVTX instrumentation. Accepts a comma separated list of "
                 "instrumentation groups. By default the NVTX instrumentation is disabled.")

#if defined(NVSHMEM_IBGDA_SUPPORT) || defined(NVSHMEM_ENV_ALL)
/** GPU-initiated communication **/
NVSHMEMI_ENV_DEF(IBGDA_NUM_DCT, int, 2, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of DCT QPs used in GPU-initiated communication transport.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_DCI, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Total number of DCI QPs used in GPU-initiated communication transport. "
                 "Set to 0 or a negative number to use automatic configuration.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_SHARED_DCI, int, 1, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of DCI QPs in the shared pool. "
                 "The rest of DCI QPs (NVSHMEM_IBGDA_NUM_DCI - NVSHMEM_IBGDA_NUM_SHARED_DCI) are "
                 "exclusively assigned. "
                 "Valid value: [1, NVSHMEM_IBGDA_NUM_DCI].")
NVSHMEMI_ENV_DEF(IBGDA_DCI_MAP_BY, string, "cta", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies how exclusive DCI QPs are assigned. "
                 "Choices are: cta, sm, warp, dct.\n\n"
                 "- cta: round-robin by CTA ID (default).\n"
                 "- sm: round-robin by SM ID.\n"
                 "- warp: round-robin by Warp ID.\n"
                 "- dct: round-robin by DCT ID.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_RC_PER_PE, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of RC QPs per peer PE used in GPU-initiated communication transport. "
                 "Set to 0 to disable RC QPs (default 0). "
                 "If set to a positive number, DCI will be used for enforcing consistency only.")
NVSHMEMI_ENV_DEF(IBGDA_RC_MAP_BY, string, "cta", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies how RC QPs are assigned. "
                 "Choices are: cta, sm, warp.\n\n"
                 "- cta: round-robin by CTA ID (default).\n"
                 "- sm: round-robin by SM ID.\n"
                 "- warp: round-robin by Warp ID.")
NVSHMEMI_ENV_DEF(IBGDA_FORCE_NIC_BUF_MEMTYPE, string, "gpumem", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Force NIC buffer memory type. Valid choices are: gpumem (default), hostmem. "
                 "For other values, use auto discovery.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_REQUESTS_IN_BATCH, int, 32, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of requests to be batched before submitting to the NIC. "
                 "It will be rounded up to the nearest power of 2. "
                 "Set to 1 for aggressive submission.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_FETCH_SLOTS_PER_DCI, int, 1024, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of internal buffer slots for fetch operations for each DCI QP. "
                 "It will be rounded up to the nearest power of 2.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_FETCH_SLOTS_PER_RC, int, 1024, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of internal buffer slots for fetch operations for each RC QP. "
                 "It will be rounded up to the nearest power of 2.")
NVSHMEMI_ENV_DEF(IB_ENABLE_IBGDA, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set to enable GPU-initiated communication transport.")
#endif

#endif
