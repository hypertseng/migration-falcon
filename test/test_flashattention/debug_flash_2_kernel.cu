#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ENABLE_NOTE_LOG 0

__global__ void flash_attn_2_fwd_f32_kernel(
    const float *Q,
    const float *K,
    const float *V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float *l,
    float *m,
    float *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3]; // l,m are in registers

    for (int i = 0; i < Tr; i++)
    {
#pragma unroll
        for (int x = 0; x < d; x++)
        {
            // load Qi to SRAM
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            // initialize O to 0
            // O[qkv_offset + (tile_size * i) + (tx * d) + x] = 0;
        }
        __syncthreads(); // such that the inner loop can use the correct Kj, Vj

        for (int j = 0; j < Tc; j++)
        {
#pragma unroll
            for (int x = 0; x < d; x++)
            {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // S[tx][y] = Sum_{x = 0}^{d-1} {Qi[tx][x] * Kj[y][x]}
            // row_m = Max_{y = 0}^{Bc-1} S[tx][y]
            float row_m = -INFINITY;
#pragma unroll
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
#pragma unroll
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            // P[tx][y] = exp(S[tx][y] - row_m)
            float row_l = 0;
#pragma unroll
            for (int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m); // here S is actually P in the paper
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;

            // compute O = diag(exp(m_prev - m_new))-1 * O_prev + P * V
            // O[tx][x] = (exp(m_prev - m_new) * O_prev[tx][x] + Sum_{y = 0}^{Bc-1} P[tx][y] * Vj[y][x]) / row_l_new
            for (int x = 0; x < d; x++)
            {
                float pv = 0;
#pragma unroll
                for (int y = 0; y < Bc; y++)
                {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / (__expf(row_m_prev - row_m_new)) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) + pv;
            }
        }
        __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop

        // Write O, l, m to HBM
#pragma unroll
        for (int x = 0; x < d; x++)
        {
            // Oi = (1 / diag(li[Tc])) * Oi[Tc]
            O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / l[lm_offset + (Br * i) + tx]) * O[qkv_offset + (tile_size * i) + (tx * d) + x];
        }
        // Li = mi[Tc] + log(li[Tc])
        l[lm_offset + (Br * i) + tx] = m[lm_offset + (Br * i) + tx] + __logf(l[lm_offset + (Br * i) + tx]);
    }
}

int main(int argc, char const *argv[])
{
    float *Q, *K, *V, *l, *m, *O;

    cudaMallocHost((void **)&Q, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMallocHost((void **)&K, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMallocHost((void **)&V, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMallocHost((void **)&l, sizeof(float) * 16 * 71 * 128);
    cudaMallocHost((void **)&m, sizeof(float) * 16 * 71 * 128);
    cudaMallocHost((void **)&O, sizeof(float) * 16 * 71 * 128 * 64);
    const int B = 16;
    const int nh = 71;
    const int N = 128;
    const int d = 64;

    cudaMemset(Q, 1.0, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMemset(K, 1.0, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMemset(V, 1.0, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMemset(l, 0, sizeof(float) * 16 * 71 * 128);
    cudaMemset(m, -INFINITY, sizeof(float) * 16 * 71 * 128);
    cudaMemset(O, 0, sizeof(float) * 16 * 71 * 128 * 64);

    // to device
    float *d_Q, *d_K, *d_V, *d_l, *d_m, *d_O;
    cudaMalloc((void **)&d_Q, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMalloc((void **)&d_K, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMalloc((void **)&d_V, sizeof(float) * 16 * 71 * 128 * 64);
    cudaMalloc((void **)&d_l, sizeof(float) * 16 * 71 * 128);
    cudaMalloc((void **)&d_m, sizeof(float) * 16 * 71 * 128);
    cudaMalloc((void **)&d_O, sizeof(float) * 16 * 71 * 128 * 64);

    cudaMemcpy(d_Q, Q, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, sizeof(float) * 16 * 71 * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, sizeof(float) * 16 * 71 * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice);

    // set block size, TODO: dynamically set block size
    const int Bc = 32;
    const int Br = 32;

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Calculate SRAM size needed per block
    const int sram_size = (2 * Bc * d * sizeof(float)) + (4 * Br * d * sizeof(float));
    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Bc);   // Bc threads per block

    flash_attn_2_fwd_f32_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, d, Tc, Tr, Bc, Br, softmax_scale, d_l, d_m, d_O);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(d_O);
    cudaFreeHost(Q);
    cudaFreeHost(K);
    cudaFreeHost(V);
    cudaFreeHost(l);
    cudaFreeHost(m);
    cudaFreeHost(O);
    return 0;
}