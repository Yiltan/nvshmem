/****
 * Copyright (c) 2014, NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * The U.S. Department of Energy funded the development of this software
 * under subcontract 7078610 with Lawrence Berkeley National Laboratory.
 *
 ****/

#include <math.h>

#include "nvshmem_internal.h"
#include "nvshmemx_error.h"
#include "nvshmemi_proxy.h"
#include "../proxy/proxy.h"

char *proxy_channel_g_buf;
char *proxy_channel_g_coalescing_buf;
uint64_t proxy_channel_g_buf_size;     /* Total size of g_buf in bytes */
uint64_t proxy_channel_g_buf_log_size; /* Total size of g_buf in bytes */

int nvshmemi_proxy_prep_minimal_state(proxy_state_t *state) {
    int *temp_global_exit_request_state;
    int *temp_global_exit_code;
    nvshmemi_timeout_t *nvshmemi_timeout_dptr;

    nvshmemi_device_state.global_exit_request_state = state->global_exit_request_state;

    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&temp_global_exit_request_state,
                                                state->global_exit_request_state, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_global_exit_code, state->global_exit_code, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&nvshmemi_timeout_dptr, state->nvshmemi_timeout, 0));

    nvshmemi_device_state.global_exit_request_state = temp_global_exit_request_state;
    nvshmemi_device_state.global_exit_code = temp_global_exit_code;
    nvshmemi_device_state.timeout = nvshmemi_timeout_dptr;

    /* Set here in case we are in an NVLink only build and don't call
     * nvshmemi_proxy_setup_device_channels*/
    nvshmemi_set_device_state(&nvshmemi_device_state);
    return 0;
}

int nvshmemi_proxy_setup_device_channels(proxy_state_t *state) {
    int status = 0;

    nvshmemi_device_state.proxy_channel_buf_size = state->channel_bufsize;
    nvshmemi_device_state.proxy_channel_buf_logsize = state->channel_bufsize_log;
    CUDA_RUNTIME_CHECK(
        cudaMalloc(&state->channels_device, sizeof(proxy_channel_t) * state->channel_count));
    INFO(NVSHMEM_PROXY, "channel buf: %p complete: %p quiet_issue: %p quiet_ack: %p",
         state->channels[0].buf, state->channels[0].complete, state->channels[0].quiet_issue,
         state->channels[0].quiet_ack);

    uint64_t *temp_buf_dptr;
    uint64_t *temp_complete_dptr;
    uint64_t *temp_quiet_issue_dptr;
    uint64_t *temp_quiet_ack_dptr;
    uint64_t *temp_cst_issue_dptr;
    uint64_t *temp_cst_ack_dptr;

    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&temp_buf_dptr, state->channels[0].buf, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_complete_dptr, state->channels[0].complete, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_quiet_issue_dptr, state->channels[0].quiet_issue, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_quiet_ack_dptr, state->channels[0].quiet_ack, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_cst_issue_dptr, state->channels[0].cst_issue, 0));
    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&temp_cst_ack_dptr, state->channels[0].cst_ack, 0));

    INFO(NVSHMEM_PROXY,
         "channel device_ptr buf: %p issue: %p complete: %p quiet_issue: %p quiet_ack: %p \n",
         temp_buf_dptr, state->channels[0].issue, temp_complete_dptr, temp_quiet_issue_dptr,
         temp_quiet_ack_dptr);

    nvshmemi_device_state.proxy_channels_buf = temp_buf_dptr;
    nvshmemi_device_state.proxy_channels_issue = state->channels[0].issue;
    nvshmemi_device_state.proxy_channels_complete = temp_complete_dptr;
    nvshmemi_device_state.proxy_channels_quiet_issue = temp_quiet_issue_dptr;
    nvshmemi_device_state.proxy_channels_quiet_ack = temp_quiet_ack_dptr;
    nvshmemi_device_state.proxy_channels_cst_issue = temp_cst_issue_dptr;
    nvshmemi_device_state.proxy_channels_cst_ack = temp_cst_ack_dptr;

    proxy_channel_g_buf_size = NUM_G_BUF_ELEMENTS * sizeof(g_elem_t);
    proxy_channel_g_buf_log_size = (uint64_t)log2((double)proxy_channel_g_buf_size);
    uint64_t *proxy_channel_g_buf_head_ptr;
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&proxy_channel_g_buf_head_ptr, sizeof(uint64_t)));
    CUDA_RUNTIME_CHECK(cudaMemset((void *)proxy_channel_g_buf_head_ptr, 0, sizeof(uint64_t)));

    uint64_t *proxy_channels_complete_local_ptr;
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&proxy_channels_complete_local_ptr, sizeof(uint64_t)));
    CUDA_RUNTIME_CHECK(cudaMemset((void *)proxy_channels_complete_local_ptr, 0, sizeof(uint64_t)));

    proxy_channel_g_buf = (char *)nvshmemi_malloc(proxy_channel_g_buf_size);
    NVSHMEMI_NULL_ERROR_JMP(proxy_channel_g_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating proxy_channel_g_buf");
    proxy_channel_g_coalescing_buf = (char *)nvshmemi_malloc(G_COALESCING_BUF_SIZE);
    NVSHMEMI_NULL_ERROR_JMP(proxy_channel_g_coalescing_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                            out, "failed allocating proxy_channel_g_coalescing_buf");

    nvshmemi_device_state.proxy_channel_g_buf_size = proxy_channel_g_buf_size;
    nvshmemi_device_state.proxy_channel_g_buf_log_size = proxy_channel_g_buf_log_size;
    nvshmemi_device_state.proxy_channel_g_buf_head_ptr = proxy_channel_g_buf_head_ptr;
    nvshmemi_device_state.proxy_channels_complete_local_ptr = proxy_channels_complete_local_ptr;
    nvshmemi_device_state.proxy_channel_g_buf = proxy_channel_g_buf;
    nvshmemi_device_state.proxy_channel_g_coalescing_buf = proxy_channel_g_coalescing_buf;
    assert(proxy_channel_g_buf_size % sizeof(g_elem_t) == 0);
    nvshmemi_set_device_state(&nvshmemi_device_state);

out:
    return status;
}
