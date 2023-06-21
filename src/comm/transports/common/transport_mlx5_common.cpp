/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_mlx5_common.h"
#include <unistd.h>

bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context) {
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));

    if (status == 0) {
        return true;
    }
    return false;
}

int nvshmemt_ib_common_query_endianness_conversion_size(uint32_t *endianness_mode,
                                                        struct ibv_context *context) {
    void *cap;
    int amo_endianness_mode;
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_ATOMIC << 1) | HCA_CAP_OPMOD_GET_MAX);
    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for atomic caps failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability);
    amo_endianness_mode = DEVX_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);
    if (amo_endianness_mode) {
        *endianness_mode = 8;
    } else {
        *endianness_mode = UINT32_MAX;
    }

out:
    return status;
}
