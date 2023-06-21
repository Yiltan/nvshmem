#include <stdio.h>
#include <cuda.h>
#include <algorithm>
#include <numeric>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cassert>
#include <vector>
#include <getopt.h>
#include <errno.h>

#include "coll_test.h"

__global__ void all_to_all(double *src, double *dst, int my_pe, int num_pes, size_t msg_size,
                           size_t num_msgs_per_dst, int *is_not_p2p) {
    int msg_id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Avoid sending too much data when num_msgs_per_dest < 1024 */
    if (msg_id < num_msgs_per_dst) {
        for (int pe_id = 0; pe_id < num_pes; pe_id++) {
            // Scattering
            int pe_dst = (my_pe + pe_id + 1) % num_pes;
            // Only send if the destination is over remote transports
            if (is_not_p2p[pe_dst]) {
                double *src_start = src + (msg_id * num_pes + pe_dst) * msg_size;
                double *dst_start = dst + (msg_id * num_pes + my_pe) * msg_size;
                nvshmem_double_put_nbi(dst_start, src_start, msg_size, pe_dst);
            }
        }
    }
}

static void print_help(void) {
    printf("Usage: {srun/mpirun/oshrun} ./alltoall_bw -n -N\n");
    printf(
        "  Runs a bandwidth benchmark that simulates the communication required by an FFT "
        "calculation. The only inputs are the minimum and maximum dimension size.\n");
    printf(
        "  You can imagine the data shared between nodes in an FFT as a 3-dimensional array with "
        "all dimensions having equal size (N).\n");
    printf(
        "  with knowledge of the FFT dimension (N) and number of GPUs (G), the number of messages "
        "sent by each GPU to each other (M) and the size of each message (S) can be calculated as "
        "follows:\n");
    printf(
        "  M = N/G, S = (Type_Size) * N^2 / G. Total Symmetric Heap requirement is equal to "
        "(Type_size) * N^3 / G.");
    printf("  Note: All *_LOG arguments should be entered as decimals from 0-63.");
    printf(
        "  -n FFT_DIM_MIN_LOG is the start of the interval of the one-dimensional FFT slice size, "
        "represented as the integer log2 of a power of 2\n");
    printf(
        "  -N FFT_DIM_MAX_LOG is the end of the interval of the one-dimensional FFT slice size, "
        "represented as the integer log2 of a power of 2\n");
}

int main(int argc, char *argv[]) {
    int ndevices;
    int mype, npes;
    int status = 0;

    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    cudaStream_t stream;

    std::vector<double> host_input;
    std::vector<double> host_output;
    std::vector<int> is_not_p2p_host;

    int *is_not_p2p = NULL;
    double *src = NULL;
    double *dst = NULL;

    size_t num_remote_dest;
    size_t size_max;
    size_t array_dim_min_log = 1;
    size_t array_dim_max_log = 0;
    size_t array_dim_min = 1;
    size_t array_dim_max = 0;
    size_t warmup = 0;
    size_t repeat = 1;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    while (1) {
        int c;
        c = getopt(argc, argv, "n:N:w:r:h");
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'n':
                array_dim_min_log = strtol(optarg, NULL, 0);
                if (array_dim_min_log > 63) {
                    print_help();
                    status = -EINVAL;
                    goto out;
                }
                array_dim_min = (1 << array_dim_min_log);
                break;
            case 'N':
                array_dim_max_log = strtol(optarg, NULL, 0);
                if (array_dim_max_log > 63) {
                    print_help();
                    status = -EINVAL;
                    goto out;
                }
                array_dim_max = (1 << array_dim_max_log);
                break;
            case 'w':
                warmup = strtol(optarg, NULL, 0);
                break;
            case 'r':
                repeat = strtol(optarg, NULL, 0);
                break;
            case 'h':
            default:
                print_help();
                goto out;
                break;
        }
    }

    if (array_dim_min < (size_t)npes) {
        fprintf(stderr,
                "array_dim_min_log is too small. With %d gpus, the minimum dimension size must be "
                "at least log2(%d)\n",
                npes, npes);
        status = -EINVAL;
        goto out;
    }

    assert(array_dim_max >= array_dim_min);
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));

    if (mype == 0) {
        printf(
            "all_to_all_rem: num_pes = %d, array_dim_min = %lu, array_dim_max = %lu, skip = %zu, "
            "repeat = %zu\n",
            npes, array_dim_min, array_dim_max, warmup, repeat);
    }

    /* Allocate the src and dest array. */
    if ((array_dim_max_log * 3) > 63) {
        fprintf(stderr,
                "array_dim_max_log (%lu) is too large and would result in overflow. Exiting.\n",
                array_dim_max_log);
        status = -EINVAL;
        goto out;
    }
    size_max = 1LLU << (array_dim_max_log * 3);
    size_max = size_max / npes;
    if (size_max > (SIZE_MAX / sizeof(double))) {
        fprintf(stderr,
                "array_dim_max_log (%lu) is too large and would result in overflow. Exiting.\n",
                array_dim_max_log);
        status = -EINVAL;
        goto out;
    }

    host_input.insert(host_input.begin(), size_max, 0.0);
    host_output.insert(host_output.begin(), size_max, 0.0);
    for (size_t i = 0; i < size_max; i++)
        host_input[i] = (double)(((size_t)mype) * 314 + (((size_t)mype) % 19) * i);

    src = (double *)nvshmem_malloc(size_max * sizeof(double));
    dst = (double *)nvshmem_malloc(size_max * sizeof(double));

    // Check who is accessible over remtoe transports vs NVLINK
    is_not_p2p_host.insert(is_not_p2p_host.begin(), npes, 1);
    if (std::getenv("NVSHMEMTEST_ALL_REMOTE")) {
        // do nothing
    } else {
        // skip comm to gpus within my node
        int my_node = (mype / ndevices);
        for (int i = my_node * ndevices; i < (my_node + 1) * ndevices; i++) {
            is_not_p2p_host[i] = 0;
        }
    }
    num_remote_dest = std::accumulate(is_not_p2p_host.begin(), is_not_p2p_host.end(), (int)0);

    CUDA_CHECK(cudaMalloc(&is_not_p2p, sizeof(int) * npes));
    CUDA_CHECK(
        cudaMemcpy(is_not_p2p, is_not_p2p_host.data(), sizeof(int) * npes, cudaMemcpyDefault));

    // Create cuda stuff
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Iterate over sizes
    if (mype == 0)
        printf(
            "         msg_size_B,   num_msg_p_dst,     num_remote_dst,     errors,        checked, "
            "         num_B,  av_time_ms, av_bw_GB_s_pe\n");
    for (size_t array_dim = array_dim_min; array_dim <= array_dim_max; array_dim *= 2) {
        size_t msg_size = (array_dim * array_dim) / npes;
        size_t num_msgs_per_dst = array_dim / npes;

        // Reset device memory
        CUDA_CHECK(cudaMemset(src, 0, size_max * sizeof(double)));
        CUDA_CHECK(cudaMemset(dst, 0, size_max * sizeof(double)));

        // Compute grid size
        size_t num_threads = std::min((size_t)1024, num_msgs_per_dst);
        size_t num_blocks = (num_msgs_per_dst + num_threads - 1) / num_threads;

        // How much data to move to GPU
        const size_t size = msg_size * num_msgs_per_dst * npes;
        const size_t msg_size_B = msg_size * sizeof(double);

        // Input -> src
        cudaMemcpy(src, host_input.data(), size * sizeof(double), cudaMemcpyDefault);

        // Skip first ones
        nvshmem_barrier_all();
        for (size_t i = 0; i < warmup; i++) {
            all_to_all<<<num_blocks, num_threads, 0, stream>>>(src, dst, mype, npes, msg_size,
                                                               num_msgs_per_dst, is_not_p2p);
            nvshmemx_barrier_all_on_stream(stream);
        }

        // Run and time
        cudaEventRecord(start);
        for (size_t i = 0; i < repeat; i++) {
            all_to_all<<<num_blocks, num_threads, 0, stream>>>(src, dst, mype, npes, msg_size,
                                                               num_msgs_per_dst, is_not_p2p);
            nvshmemx_barrier_all_on_stream(stream);
        }
        cudaEventRecord(stop);

        // Measure and display time
        cudaEventSynchronize(stop);
        cudaStreamSynchronize(stream);
        float time_ms = 0;
        cudaEventElapsedTime(&time_ms, start, stop);
        float average_time_ms = time_ms / repeat;
        size_t num_B = msg_size * num_msgs_per_dst * num_remote_dest * sizeof(double);
        float average_bw_GB_s = (num_B / 1e9) / (average_time_ms * 1e-3);

        // Check correctness
        size_t errors = 0;
        size_t checked = 0;
        if (!std::getenv("NVSHMEMTEST_SKIP_CHECK")) {
            cudaMemcpy(host_output.data(), dst, size * sizeof(double), cudaMemcpyDefault);
            for (size_t i = 0; i < size; i += 1) {
                size_t chunk_id = i / (npes * msg_size);
                size_t src_pe = (i % (npes * msg_size)) / msg_size;
                size_t chunk_ii = (i % (npes * msg_size)) % msg_size;
                size_t src_ii = mype * msg_size + chunk_id * npes * msg_size + chunk_ii;
                double ref = 0.0;
                if (is_not_p2p_host[src_pe]) {
                    ref = (double)(((size_t)src_pe) * 314 + (((size_t)src_pe) % 19) * src_ii);
                }
                if (ref != host_output[i]) {
                    errors++;
                }
                checked++;
            }
        }

        if (mype == 0)
            printf("%14zu,  %14zu, %14zu, %14zu, %14zu, %14zu,  %5.3e,   %5.3e\n", msg_size_B,
                   num_msgs_per_dst, num_remote_dest, errors, checked, num_B, average_time_ms,
                   average_bw_GB_s);
    }

out:
    if (is_not_p2p) {
        cudaFree(is_not_p2p);
    }
    if (stream) {
        cudaStreamDestroy(stream);
    }
    if (start) {
        cudaEventDestroy(start);
    }
    if (stop) {
        cudaEventDestroy(stop);
    }

    if (src) {
        nvshmem_free(src);
    }
    if (dst) {
        nvshmem_free(dst);
    }

    finalize_wrapper();
    return status;
}
