/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <algorithm>
#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <stdio.h>
#include "nvshmemx_error.h"
#include "util.h"

inline int nvshmemi_minv(int *vec, int count) {
    int minval = INT_MAX;
    for (int i = 0; i < count; i++) {
        if (vec[i] < minval) {
            minval = vec[i];
        }
    }
    return minval;
}

inline int nvshmemi_maxv(int *vec, int count) {
    int maxval = INT_MIN;
    for (int i = 0; i < count; i++) {
        if (vec[i] > maxval) {
            maxval = vec[i];
        }
    }
    return maxval;
}

int nvshmemi_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize) {
    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int status = 0;

    multiProcessorCount = nvshmemi_state->cu_dev_attrib.multi_processor_count;
    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed \n");

    // XXX: Returns maximum supported grid (including 0) on associated GPU
    *gridsize = maxBlocksSM * multiProcessorCount;  // XXX:caller chooses dimension of grid

out:
    return status;
}

int nvshmemi_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream) {
    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int gridSize = -1;
    int launchFailed = 1;
    int status = 0;

    // XXX: Supports the user passing a non-zero grid but of differing size across ranks
    if (gridDims.x == 0 && gridDims.y == 0 && gridDims.z == 0) {
        gridSize = 0;
    } else if (gridDims.x != 0 && gridDims.y != 0 && gridDims.z != 0) {
        gridSize = gridDims.x * gridDims.y * gridDims.z;
    }  // else
       // some but not all grid dim being 0 is illegal
       // XXX: if some ranks pass an illegal grid, others error out

    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed \n");

    multiProcessorCount = nvshmemi_state->cu_dev_attrib.multi_processor_count;
    INFO(NVSHMEM_COLL, "[%d] SM count %d  CTA/SM count %d", nvshmemi_state->mype,
         multiProcessorCount, maxBlocksSM);
    if (gridSize == 0) { /*XXX : auto sizing */
        // two alternatives - run the minimum supported grid (>0) on all GPUs (global communication
        // needed) or run the maximum supported grid on each GPU (local decision)
        // XXX: Launches maximum supported grid (>0) on associated GPU
        if (maxBlocksSM > 0) { /*Launch will work only if all GPUs can run at least one CTA*/
            launchFailed = 0;
        }
        gridDims.x = maxBlocksSM * multiProcessorCount;
        gridDims.y = 1;
        gridDims.z = 1;
    } else if (gridSize > 0) { /* XXX : legal grid is provided by user*/
        if ((maxBlocksSM > 0) && (gridSize <= maxBlocksSM * multiProcessorCount)) { /*Works*/
            launchFailed = 0;
        }
    }
#ifdef _NVSHMEM_DEBUG
    INFO(NVSHMEM_COLL, "nvshmemi_maxv allgather target %p source %p nbytes %ld", &launchFailed,
         nvshmemi_state->scratch, sizeof(int));
    status = nvshmemi_boot_handle.allgather((void *)&launchFailed, (void *)nvshmemi_state->scratch,
                                            sizeof(int), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                          "allgather of launch capability failed \n");

    launchFailed = nvshmemi_maxv(nvshmemi_state->scratch, nvshmemi_state->npes);
#endif
    /* TODO: make it obvious we aren't going to complete this call from this thread. Possibly global
     * exit? */
    NVSHMEMI_NZ_ERROR_JMP(launchFailed, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                          "One or more PEs cannot launch \n");

    CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(nvshmemi_state->claunch_params.begin_event, stream),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(nvshmemi_state->claunch_params.stream,
                                                nvshmemi_state->claunch_params.begin_event, 0),
                            status, out);

    if (nvshmemi_state->cu_dev_attrib.cooperative_launch) {
        status = cudaLaunchCooperativeKernel(func, gridDims, blockDims, args, sharedMem,
                                             nvshmemi_state->claunch_params.stream);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                              "Cooperative kernel launch failed \n");
    } else {
        status = cudaLaunchKernel(func, gridDims, blockDims, args, sharedMem,
                                  nvshmemi_state->claunch_params.stream);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                              "Kernel launch failed \n");
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(nvshmemi_state->claunch_params.end_event,
                                            nvshmemi_state->claunch_params.stream),
                            status, out);

    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamWaitEvent(stream, nvshmemi_state->claunch_params.end_event, 0), status, out);

out:
    return status;
}
