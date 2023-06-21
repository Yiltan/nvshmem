/****
 * Copyright (c) 2016-2021, NVIDIA Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>
#include <inttypes.h>
#include <errno.h>

#include "nvshmem_nvtx.hpp"
#include "util.h"
#include "nvshmem_internal.h"
#include "nvshmem.h"

struct nvshmemi_options_s nvshmemi_options;

int nvshmemi_options_init(void) { return nvshmemi_env_options_init(&nvshmemi_options); }

static void nvshmemi_options_print_heading(const char *h, int style) {
    switch (style) {
        case NVSHMEMI_OPTIONS_STYLE_INFO:
            printf("%s:\n", h);
            break;
        case NVSHMEMI_OPTIONS_STYLE_RST:
            printf("%s\n", h);
            for (const char *c = h; *c != '\0'; c++) putchar('~');
            printf("\n\n");
            break;
        default:
            assert(0);  // FIXME
    }
}

#define NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, DESIRED_CAT, STYLE) \
    if (CATEGORY == DESIRED_CAT) {                                                                \
        switch (STYLE) {                                                                          \
            char *desc_wrapped;                                                                   \
            case NVSHMEMI_OPTIONS_STYLE_INFO:                                                     \
                desc_wrapped = nvshmemu_wrap(SHORT_DESC, NVSHMEMI_WRAPLEN, "\t", 1);              \
                printf("  NVSHMEM_%-20s " NVSHPRI_##KIND " (type: %s, default: " NVSHPRI_##KIND   \
                       ")\n\t%s\n",                                                               \
                       #NAME, NVSHFMT_##KIND(nvshmemi_options.NAME), #KIND,                       \
                       NVSHFMT_##KIND(DEFAULT), desc_wrapped);                                    \
                free(desc_wrapped);                                                               \
                break;                                                                            \
            case NVSHMEMI_OPTIONS_STYLE_RST:                                                      \
                desc_wrapped = nvshmemu_wrap(SHORT_DESC, NVSHMEMI_WRAPLEN, NULL, 0);              \
                printf(".. c:var:: NVSHMEM_%s\n", #NAME);                                         \
                printf("\n");                                                                     \
                printf("| *Type: %s*\n", #KIND);                                                  \
                printf("| *Default: " NVSHPRI_##KIND "*\n", NVSHFMT_##KIND(DEFAULT));             \
                printf("\n");                                                                     \
                printf("%s\n", desc_wrapped);                                                     \
                printf("\n");                                                                     \
                free(desc_wrapped);                                                               \
                break;                                                                            \
            default:                                                                              \
                assert(0); /* FIXME */                                                            \
        }                                                                                         \
    }

void nvshmemi_options_print(int style) {
    nvshmemi_options_print_heading("Standard options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_OPENSHMEM, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Bootstrap options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Additional options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                               \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_OTHER, \
                               style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Collectives options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_COLLECTIVES, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Transport options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_TRANSPORT, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    if (nvshmemi_options.INFO_HIDDEN) {
        nvshmemi_options_print_heading("Hidden options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                                \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_HIDDEN, \
                               style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
        printf("\n");
    }

#ifndef NVTX_DISABLE
    if (nvshmemi_options.NVTX) {
        nvshmemi_options_print_heading("NVTX options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                              \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_NVTX, \
                               style)
#include "env_defs.h"

        if (style == NVSHMEMI_OPTIONS_STYLE_RST) printf(".. code-block:: none\n\n");

        nvshmem_nvtx_print_options();
#undef NVSHMEMI_ENV_DEF
    }
#endif /* !NVTX_DISABLE */

    printf("\n");
}
