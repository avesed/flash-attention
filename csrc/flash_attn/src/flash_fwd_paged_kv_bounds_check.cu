/******************************************************************************
 * Device-side bounds check for mha_fwd_kvcache with a paged KV cache.
 ******************************************************************************/

// The check must fire in release builds too, which define NDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

#include "namespace_config.h"
#include "cuda_check.h"

namespace FLASH_NAMESPACE {

// One thread per sequence: seqlens_k[b] + seqlen_knew must fit in the pages its block_table row can
// address. A violation is a device-side assert, which stops the stream before the attention kernel
// (queued after this one on the same stream) reads block_table out of bounds.
__global__ void paged_kv_seqlens_bound_check_kernel(const int *__restrict__ seqlens_k, int batch_size,
                                                    int seqlen_knew, int capacity) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        const int seqlen_k = seqlens_k[b] + seqlen_knew;
        if (seqlen_k > capacity) {
            printf("FlashAttention paged KV cache: seqlens_k[%d] + seqlen_knew (= %d) exceeds the capacity "
                   "addressable by block_table (max_num_blocks_per_seq * page_block_size = %d)\n",
                   b, seqlen_k, capacity);
        }
        assert(seqlen_k <= capacity && "paged KV: seqlens_k exceeds block_table capacity");
    }
}

void run_paged_kv_seqlens_bound_check(const int *seqlens_k, int batch_size, int seqlen_knew, int capacity,
                                      cudaStream_t stream) {
    if (batch_size <= 0) { return; }
    constexpr int kThreads = 128;
    paged_kv_seqlens_bound_check_kernel<<<(batch_size + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        seqlens_k, batch_size, seqlen_knew, capacity);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace FLASH_NAMESPACE
