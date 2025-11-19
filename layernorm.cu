#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================================
// Multi-CTA Norm forward - CUDA
// Supports two compile-time selectable modes (default = LayerNorm):
//   NORM_MODE=0 -> LayerNorm:   y = (x - mean) / sqrt(var + eps) * weight + bias
//   NORM_MODE=1 -> RMSNorm:     y = x / sqrt(mean(x^2) + eps)  * weight + bias
//
// Build-time selection (examples):
//   - keep default LayerNorm:
//       extra_cuda_cflags: ["-O3", "--use_fast_math"]
//   - enable RMSNorm:
//       extra_cuda_cflags: ["-O3", "--use_fast_math", "-DNORM_MODE=1"]
//
// This preserves the optimized memory/compute path while enabling alternative
// norm variants for models trained with them.
// ============================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef NORM_MODE
#define NORM_MODE 0
#endif

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffffu;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}

// Kernel 1: partial reduction per (group, tile)
__global__ void ln_partial_reduce_kernel(
    const float* __restrict__ x,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    int M,                  // number of groups
    int D,                  // elements per group
    int group_cta)          // CTAs assigned per group
{
    int bid = blockIdx.x;
    int row = bid / group_cta;
    if (row >= M) return;
    int tile = bid - row * group_cta;

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ float shared[];
    float* s_sum = shared;                 // up to 32 warps
    float* s_sumsq = shared + 32;          // up to 32 warps

    // Compute chunk range for this tile
    int chunk_size = (D + group_cta - 1) / group_cta;
    int start = tile * chunk_size;
    int end = start + chunk_size;
    if (end > D) end = D;

    size_t base = (size_t)row * (size_t)D;

    float t_sum = 0.0f;
    float t_sumsq = 0.0f;

    // Unrolled, grid-stride loop for better ILP
    int i0 = start + tid;
    int stride = blockDim.x;
    for (; i0 + 3 * stride < end; i0 += 4 * stride) {
        float v0 = x[base + i0];
        float v1 = x[base + i0 + stride];
        float v2 = x[base + i0 + 2 * stride];
        float v3 = x[base + i0 + 3 * stride];
        t_sum += v0 + v1 + v2 + v3;
        t_sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for (; i0 < end; i0 += stride) {
        float v = x[base + i0];
        t_sum += v;
        t_sumsq += v * v;
    }

    float w_sum = warp_reduce_sum(t_sum);
    float w_sumsq = warp_reduce_sum(t_sumsq);

    if (lane == 0) {
        s_sum[warp_id] = w_sum;
        s_sumsq[warp_id] = w_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v_sum = (tid < nwarps) ? s_sum[tid] : 0.0f;
        float v_sumsq = (tid < nwarps) ? s_sumsq[tid] : 0.0f;
        v_sum = warp_reduce_sum(v_sum);
        v_sumsq = warp_reduce_sum(v_sumsq);
        if (lane == 0) {
            partial_sum[bid] = v_sum;
            partial_sumsq[bid] = v_sumsq;
        }
    }
}

// Kernel 2: reduce partial tiles to final stats per group
__global__ void ln_reduce_stats_kernel(
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_sumsq,
    float* __restrict__ out_mean,
    float* __restrict__ out_invstd,
    int M,
    int D,
    int group_cta,
    float eps)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ float shared[];
    float* s_sum = shared;                 // up to 32 warps
    float* s_sumsq = shared + 32;          // up to 32 warps

    // Accumulate across tiles for this group
    float t_sum = 0.0f;
    float t_sumsq = 0.0f;
    int base = row * group_cta;

    int stride = blockDim.x;
    int t = tid;
    for (; t + 3 * stride < group_cta; t += 4 * stride) {
        float s0 = partial_sum[base + t];
        float s1 = partial_sum[base + t + stride];
        float s2 = partial_sum[base + t + 2 * stride];
        float s3 = partial_sum[base + t + 3 * stride];
        float q0 = partial_sumsq[base + t];
        float q1 = partial_sumsq[base + t + stride];
        float q2 = partial_sumsq[base + t + 2 * stride];
        float q3 = partial_sumsq[base + t + 3 * stride];
        t_sum += s0 + s1 + s2 + s3;
        t_sumsq += q0 + q1 + q2 + q3;
    }
    for (; t < group_cta; t += stride) {
        t_sum += partial_sum[base + t];
        t_sumsq += partial_sumsq[base + t];
    }

    float w_sum = warp_reduce_sum(t_sum);
    float w_sumsq = warp_reduce_sum(t_sumsq);

    if (lane == 0) {
        s_sum[warp_id] = w_sum;
        s_sumsq[warp_id] = w_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v_sum = (tid < nwarps) ? s_sum[tid] : 0.0f;
        float v_sumsq = (tid < nwarps) ? s_sumsq[tid] : 0.0f;
        v_sum = warp_reduce_sum(v_sum);
        v_sumsq = warp_reduce_sum(v_sumsq);
        if (lane == 0) {
#if (NORM_MODE == 0)
            // LayerNorm statistics
            float mean = v_sum / (float)D;
            float var = v_sumsq / (float)D - mean * mean;
            if (var < 0.0f) var = 0.0f;
            float inv_std = rsqrtf(var + eps);
            out_mean[row] = mean;
            out_invstd[row] = inv_std;
#else
            // RMSNorm statistics: mean kept at 0, inv_scale computed from mean(x^2)
            float rms2 = v_sumsq / (float)D;
            float inv_scale = rsqrtf(rms2 + eps);
            out_mean[row] = 0.0f;
            out_invstd[row] = inv_scale;
#endif
        }
    }
}

// Kernel 3: normalize and affine in parallel across CTAs
__global__ void ln_normalize_affine_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ y,
    int M,
    int D,
    int group_cta)
{
    int bid = blockIdx.x;
    int row = bid / group_cta;
    if (row >= M) return;
    int tile = bid - row * group_cta;

    int tid = threadIdx.x;

    int chunk_size = (D + group_cta - 1) / group_cta;
    int start = tile * chunk_size;
    int end = start + chunk_size;
    if (end > D) end = D;

    float m = mean[row];
    float s = invstd[row];
    size_t base = (size_t)row * (size_t)D;

    int i0 = start + tid;
    int stride = blockDim.x;

    // Unrolled normalization + affine
    for (; i0 + 3 * stride < end; i0 += 4 * stride) {
        int i1 = i0 + stride;
        int i2 = i0 + 2 * stride;
        int i3 = i0 + 3 * stride;

        float v0 = x[base + i0];
        float v1 = x[base + i1];
        float v2 = x[base + i2];
        float v3 = x[base + i3];

        float n0 = (v0 - m) * s;
        float n1 = (v1 - m) * s;
        float n2 = (v2 - m) * s;
        float n3 = (v3 - m) * s;

        float w0 = weight[i0];
        float w1 = weight[i1];
        float w2 = weight[i2];
        float w3 = weight[i3];

        float b0 = bias[i0];
        float b1 = bias[i1];
        float b2 = bias[i2];
        float b3 = bias[i3];

        y[base + i0] = n0 * w0 + b0;
        y[base + i1] = n1 * w1 + b1;
        y[base + i2] = n2 * w2 + b2;
        y[base + i3] = n3 * w3 + b3;
    }
    for (; i0 < end; i0 += stride) {
        float v = x[base + i0];
        float n = (v - m) * s;
        float w = weight[i0];
        float b = bias[i0];
        y[base + i0] = n * w + b;
    }
}

// Host forward function - signature must match Python forward_fn exactly
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int64_t D64 = weight.numel();
    TORCH_CHECK(D64 > 0, "weight/bias must have non-zero numel");
    TORCH_CHECK(x.numel() % D64 == 0, "x.numel() must be divisible by weight.numel()");

    int D = (int)D64;
    int M = (int)(x.numel() / D64);

    auto y = torch::empty_like(x);

    // Heuristics for CTA partitioning and threads
    // Aim for around 4k-8k total blocks for large problems
    int group_cta_by_size = (D + 16384 - 1) / 16384;  // ~16k elems per CTA
    int target_blocks = 6144; // slightly higher to better feed H100
    int group_cta_by_target = (target_blocks + M - 1) / M;
    int group_cta = group_cta_by_size;
    if (group_cta < group_cta_by_target) group_cta = group_cta_by_target;
    if (group_cta < 1) group_cta = 1;
    if (group_cta > 2048) group_cta = 2048; // guard memory and grid size

    int chunk_size = (D + group_cta - 1) / group_cta;

    // Choose threads based on chunk size
    int threads = 256;
    if (chunk_size >= 32768) threads = 512;
    else if (chunk_size >= 8192) threads = 256;
    else if (chunk_size >= 2048) threads = 128;
    else threads = 64;
    if (threads < 64) threads = 64;

    // Temporary buffers
    auto opts = x.options().dtype(torch::kFloat32);
    auto partial_sum = torch::empty({(int64_t)M * (int64_t)group_cta}, opts);
    auto partial_sumsq = torch::empty({(int64_t)M * (int64_t)group_cta}, opts);
    auto mean = torch::empty({M}, opts);
    auto invstd = torch::empty({M}, opts);

    size_t shmem_reduce = (32 + 32) * sizeof(float);

    // Kernel 1: partial reductions
    dim3 blocks1((unsigned int)(M * group_cta));
    ln_partial_reduce_kernel<<<blocks1, threads, shmem_reduce>>>(
        x.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sumsq.data_ptr<float>(),
        M, D, group_cta
    );

    // Kernel 2: reduce to final stats
    dim3 blocks2((unsigned int)M);
    int threads2 = 256;
    ln_reduce_stats_kernel<<<blocks2, threads2, shmem_reduce>>>(
        partial_sum.data_ptr<float>(),
        partial_sumsq.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        M, D, group_cta, (float)eps
    );

    // Kernel 3: normalize and affine
    ln_normalize_affine_kernel<<<blocks1, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        y.data_ptr<float>(),
        M, D, group_cta
    );

    return y;
}

// PyBind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA kernel implementation");
}
