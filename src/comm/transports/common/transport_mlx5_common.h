/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_MLX5_COMMON_H
#define _TRANSPORT_MLX5_COMMON_H

#include "transport_common.h"
#include "infiniband/verbs.h"
#include "mlx5_ifc.h"
#include "mlx5_prm.h"
#include "infiniband/mlx5dv.h"

bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context);
int nvshmemt_ib_common_query_endianness_conversion_size(uint32_t *endianness_mode,
                                                        struct ibv_context *context);

#endif