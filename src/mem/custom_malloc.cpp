/*NVSHMEM specific macros
 * only using mspaces
 * single space
 * no mmap
 * no thread safety
 * only linux*/

#include <stdint.h>
#include <stdio.h>
#include "dlmalloc.h"
#include "cuda.h"
#include "util.h"
#include "nvshmem_internal.h"
#include <map>
#include <numeric>
using namespace std;

#define MALLOC_ALIGNMENT ((size_t)512U)
#define SIZE_T_ONE ((size_t)1)
#define MALLOC_ALIGNMENT ((size_t)512U)
#define CHUNK_ALIGN_MASK (MALLOC_ALIGNMENT - SIZE_T_ONE)

#define align_request(req) (((req) + CHUNK_ALIGN_MASK) & ~CHUNK_ALIGN_MASK)
/* the number of bytes to offset an address to align it */
#define align_offset(A)                    \
    ((((size_t)(A)&CHUNK_ALIGN_MASK) == 0) \
         ? 0                               \
         : ((MALLOC_ALIGNMENT - ((size_t)(A)&CHUNK_ALIGN_MASK)) & CHUNK_ALIGN_MASK))

/* free_chunks_start is mapping of start address of each free chunk to size of that chunk */
/* free_chunks_end is mapping of end address of each free chunk to size of that chunk */
map<void *, size_t> free_chunks_start, free_chunks_end;
/* in_use_cunks is a mapping of each in use chunks start address to size of the chunk */
map<void *, size_t> inuse_chunks;
static size_t total_size = 0; /* size of total space managed by mspace */

#ifdef _NVSHMEM_DEBUG
static size_t get_total_size(std::map<void *, size_t> chunk_map) {
    size_t sum = 0;
    for (map<void *, size_t>::iterator it = chunk_map.begin(); it != chunk_map.end(); it++) {
        sum += it->second;
    }
    return sum;
}

#define ASSERT_CORRECTNESS                                                                         \
    INFO(NVSHMEM_MEM,                                                                              \
         "get_total_size(free_chunks_start): %zu, get_total_size(in_use_cunks): %zu, total_size: " \
         "%zu\n",                                                                                  \
         get_total_size(free_chunks_start), get_total_size(inuse_chunks), total_size);             \
    assert(get_total_size(free_chunks_start) == get_total_size(free_chunks_end));                  \
    assert(get_total_size(free_chunks_start) + get_total_size(inuse_chunks) == total_size);
#else
#define ASSERT_CORRECTNESS
#endif

void mspace_print(mspace msp) {
    printf("free_chunks_start: ");
    for (map<void *, size_t>::iterator it = free_chunks_start.begin();
         it != free_chunks_start.end(); it++) {
        printf("(%p, %zu) ", it->first, it->second);
    }
    printf("\n");

    printf("free_chunks_end: ");
    for (map<void *, size_t>::iterator it = free_chunks_end.begin(); it != free_chunks_end.end();
         it++) {
        printf("(%p, %zu) ", it->first, it->second);
    }
    printf("\n");

    printf("inuse_chunks: ");
    for (map<void *, size_t>::iterator it = inuse_chunks.begin(); it != inuse_chunks.end(); it++) {
        printf("(%p, %zu) ", it->first, it->second);
    }
    printf("\n");
}

mspace create_mspace_with_base(void *base, size_t capacity, int locked) {
    char *start_addr = (char *)base;
    size_t offset = align_offset(start_addr);
    start_addr += offset;
    capacity -= offset;
    if (capacity > 0) {
        char *end_addr = start_addr + capacity;

        free_chunks_start[start_addr] = capacity;
        free_chunks_end[end_addr] = capacity;
        total_size = capacity;
    }
    // mspace_print(base);
    ASSERT_CORRECTNESS
    return &free_chunks_start;
}

void mspace_add_free_chunk(mspace msp, char *base, size_t capacity) {
    bool merged = 0;
    /* check if previous chunk is free */
    if (free_chunks_end.find(base) != free_chunks_end.end()) {
        size_t psize = free_chunks_end[base];
        free_chunks_end.erase(base);
        free_chunks_end[base + capacity] = capacity + psize;
        base = base - psize;
        capacity += psize;
        free_chunks_start[base] = capacity;
        merged = 1;
    }
    /* check if next chunk is free */
    if (free_chunks_start.find(base + capacity) != free_chunks_start.end()) {
        size_t nsize = free_chunks_start[base + capacity];
        free_chunks_end.erase(base + capacity);
        free_chunks_start.erase(base + capacity);
        free_chunks_start[base] = capacity + nsize;
        free_chunks_end[base + capacity + nsize] = capacity + nsize;
        merged = 1;
    }

    if (!merged) {
        free_chunks_start[base] = capacity;
        free_chunks_end[base + capacity] = capacity;
    }
    ASSERT_CORRECTNESS
}

void mspace_add_new_chunk(mspace msp, void *base, size_t capacity) {
    total_size += capacity;
    mspace_add_free_chunk(msp, (char *)base, capacity);
}

size_t destroy_mspace(mspace msp) {
    free_chunks_start.clear();
    free_chunks_end.clear();
    inuse_chunks.clear();
    total_size = 0;

    return 0;
}

int mspace_track_large_chunks(mspace msp, int enable) { return 0; }

void *mspace_malloc(mspace msp, size_t bytes) {
    INFO(NVSHMEM_MEM, "mspace_malloc called with %zu bytes", bytes);
    if (bytes == 0) return NULL;
    bytes = align_request(bytes);
    for (map<void *, size_t>::iterator it = free_chunks_start.begin();
         it != free_chunks_start.end(); it++) {
        if (it->second >= bytes) {
            INFO(NVSHMEM_MEM, "free chunk with size = %zu bytes found", it->second);
            char *start_addr = (char *)it->first;
            size_t rsize = it->second - bytes;
            if (rsize > 0) {
                free_chunks_start[start_addr + bytes] = rsize;
                free_chunks_end[start_addr + it->second] = rsize;
                free_chunks_start.erase(start_addr);
            } else {
                free_chunks_end.erase(start_addr + it->second);
                free_chunks_start.erase(start_addr);
            }
            inuse_chunks[start_addr] = bytes;
            ASSERT_CORRECTNESS
            return start_addr;
        }
    }
    return NULL;
}

void mspace_free(mspace msp, void *mem) {
    INFO(NVSHMEM_MEM, "mspace_free called on %p", mem);
    if (inuse_chunks.find(mem) == inuse_chunks.end()) {
        printf("Free called on an invalid pointer\n");
        exit(-1);
    }
    size_t bytes = inuse_chunks[mem];
    inuse_chunks.erase(mem);

    mspace_add_free_chunk(msp, (char *)mem, bytes);
    ASSERT_CORRECTNESS
}

void *mspace_calloc(mspace msp, size_t n_elements, size_t elem_size) {
    INFO(NVSHMEM_MEM, "mspace_calloc called with n_elements = %zu, elem_size = %zu", n_elements,
         elem_size);
    size_t bytes = n_elements * elem_size;
    void *ptr = mspace_malloc(msp, bytes);
    if (ptr) CUDA_RUNTIME_CHECK(cudaMemset(ptr, 0, bytes));
    ASSERT_CORRECTNESS
    return ptr;
}

void *mspace_memalign(mspace msp, size_t alignment, size_t bytes) {
    INFO(NVSHMEM_MEM, "mspace_memalign called with alignment = %zu, bytes = %zu", alignment, bytes);
    assert((alignment % sizeof(void *)) == 0 && ((alignment & (alignment - 1)) == 0));
    /* Request bytes + alignment for simplicity */
    bytes += alignment;
    char *ptr = (char *)mspace_malloc(msp, bytes);
    if (!ptr) return NULL;
    char *ret_ptr = (char *)(alignment * (((uint64_t)ptr + (alignment - 1)) / alignment));
    if (ret_ptr - ptr) {
        inuse_chunks[ret_ptr] = inuse_chunks[ptr] - (ret_ptr - ptr);
        inuse_chunks.erase(ptr);
        mspace_add_free_chunk(msp, ptr, ret_ptr - ptr);
    }
    ASSERT_CORRECTNESS
    return ret_ptr;
}

void *mspace_realloc(mspace msp, void *ptr, size_t size) {
    INFO(NVSHMEM_MEM, "mspace_realloc called with ptr = %p, size = %zu", ptr, size);
    size = align_request(size);
    size_t current_size = inuse_chunks[ptr];
    if (size < current_size) {
        inuse_chunks[ptr] = size;
        mspace_add_free_chunk(msp, (char *)ptr + size, current_size - size);
        ASSERT_CORRECTNESS
        return ptr;
    } else if (size > current_size) {
        if (free_chunks_start.find((char *)ptr + current_size) != free_chunks_start.end()) {
            size_t chunk_size = free_chunks_start[(char *)ptr + current_size];
            if (current_size + chunk_size >= size) {
                inuse_chunks[ptr] = size;
                free_chunks_start.erase((char *)ptr + current_size);
                free_chunks_end.erase((char *)ptr + current_size + chunk_size);
                if (current_size + chunk_size > size)
                    mspace_add_free_chunk(msp, (char *)ptr + size, size - current_size);
                ASSERT_CORRECTNESS
                return ptr;
            }
        }
        void *new_ptr = mspace_malloc(msp, size);
        if (new_ptr == NULL) return NULL;
        CUDA_RUNTIME_CHECK(cudaMemcpy(new_ptr, ptr, size, cudaMemcpyDeviceToDevice));
        inuse_chunks.erase(ptr);
        mspace_add_free_chunk(msp, (char *)ptr, current_size);
        ASSERT_CORRECTNESS
        return new_ptr;
    } else {
        return ptr;
    }
}
