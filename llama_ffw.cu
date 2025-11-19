#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ============================================================================
// Device helpers
// ============================================================================
__device__ __forceinline__ float silu_f(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return x * s;
}

// Fused pointwise kernel with robust 2D grid mapping:
// up[i] = SiLU(gate[i]) * up[i]  for float32 tensors
__global__ void silu_mul_inplace_f32(
    const float* __restrict__ gate,
    float* __restrict__ up,
    long long total_elems)
{
    long long block_linear = (long long)blockIdx.y * (long long)gridDim.x + (long long)blockIdx.x;
    long long idx = block_linear * (long long)blockDim.x + (long long)threadIdx.x;
    long long stride = ((long long)gridDim.x * (long long)gridDim.y) * (long long)blockDim.x;

    for (long long i = idx; i < total_elems; i += stride) {
        float g = gate[i];
        float u = up[i];
        up[i] = silu_f(g) * u;
    }
}

// ============================================================================
// Host forward() - concurrent input GEMMs on separate streams with events,
// fused pointwise and final GEMM on a dedicated third non-default stream.
// ============================================================================
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor gate_proj,
    torch::Tensor up_proj,
    torch::Tensor down_proj)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gate_proj.is_cuda() && up_proj.is_cuda() && down_proj.is_cuda(),
                "All weight tensors must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Expected x to be float32");
    TORCH_CHECK(gate_proj.scalar_type() == at::kFloat &&
                up_proj.scalar_type() == at::kFloat &&
                down_proj.scalar_type() == at::kFloat, "Weights must be float32");

    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, T, F)");
    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t K = x.size(2); // num_features
    int64_t M = B * T;

    TORCH_CHECK(gate_proj.dim() == 2 && up_proj.dim() == 2 && down_proj.dim() == 2,
                "All projection matrices must be 2D");
    int64_t N = gate_proj.size(0);
    TORCH_CHECK(gate_proj.size(1) == K, "gate_proj must have shape [N, K]");
    TORCH_CHECK(up_proj.size(0) == N && up_proj.size(1) == K, "up_proj must match gate_proj shape");
    TORCH_CHECK(down_proj.size(0) == K && down_proj.size(1) == N,
                "down_proj must have shape [K, N]");

    // Ensure input is contiguous and flatten batch/tokens
    auto x2d = x.contiguous().view({M, K});

    // Preallocate outputs for GEMMs to avoid allocator overhead
    auto gate = at::empty({M, N}, x.options());
    auto up   = at::empty({M, N}, x.options());
    auto y2d  = at::empty({M, K}, x.options());

    // Create two non-default streams for concurrent GEMMs and a third for dependent work
    int dev_index = x.get_device();
    auto s1 = at::cuda::getStreamFromPool(false, dev_index);
    auto s2 = at::cuda::getStreamFromPool(false, dev_index);
    auto s3 = at::cuda::getStreamFromPool(false, dev_index); // dedicated stream for fused pointwise + final GEMM

    // Launch input GEMMs concurrently on their respective streams
    {
        c10::cuda::CUDAStreamGuard guard(s1);
        at::mm_out(gate, x2d, gate_proj.t()); // [M,K] @ [K,N] -> [M,N]
    }
    {
        c10::cuda::CUDAStreamGuard guard(s2);
        at::mm_out(up,   x2d, up_proj.t());   // [M,K] @ [K,N] -> [M,N]
    }

    // Record events on the GEMM streams
    cudaEvent_t ev1, ev2;
    cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming);
    cudaEventRecord(ev1, s1.stream());
    cudaEventRecord(ev2, s2.stream());

    // Make s3 wait for both GEMMs to finish (do not block the default stream)
    cudaStreamWaitEvent(s3.stream(), ev1, 0);
    cudaStreamWaitEvent(s3.stream(), ev2, 0);

    // Events no longer needed after waits are enqueued
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);

    // Fused SiLU(gate) * up in-place on s3
    const int threads = 256;
    long long total = (long long)M * (long long)N;

    long long blocks_total = (total + threads - 1) / threads;
    if (blocks_total < 1) blocks_total = 1;

    int grid_x = (int)((blocks_total < 65535LL) ? blocks_total : 65535LL);
    if (grid_x < 1) grid_x = 1;
    long long grid_y_ll = (blocks_total + grid_x - 1) / grid_x;
    int grid_y = (int)((grid_y_ll < 65535LL) ? grid_y_ll : 65535LL);
    if (grid_y < 1) grid_y = 1;

    dim3 grid((unsigned)grid_x, (unsigned)grid_y, 1u);

    silu_mul_inplace_f32<<<grid, threads, 0, s3.stream()>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        total
    );

    // Final projection on s3: y2d = up @ down_proj^T => [M x K]
    {
        c10::cuda::CUDAStreamGuard guard(s3);
        at::mm_out(y2d, up, down_proj.t());
    }

    // Reshape back to [B, T, K]
    auto y = y2d.view({B, T, K});
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA kernel implementation");
}
