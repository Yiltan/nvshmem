/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef BOOTSTRAP_UTIL_H
#define BOOTSTRAP_UTIL_H

#include <stdio.h>

#define BOOTSTRAP_ERROR_PRINT(...)                                       \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
    } while (0)

#define BOOTSTRAP_NE_ERROR_JMP(status, expected, err, label, ...)                       \
    do {                                                                                \
        if (status != expected) {                                                       \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define BOOTSTRAP_NZ_ERROR_JMP(status, err, label, ...)                                 \
    do {                                                                                \
        if (status != 0) {                                                              \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#endif

#define BOOTSTRAP_NULL_ERROR_JMP(var, status, err, label, ...)         \
    do {                                                               \
        if (var == NULL) {                                             \
            fprintf(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                              \
            status = err;                                              \
            goto label;                                                \
        }                                                              \
    } while (0)
