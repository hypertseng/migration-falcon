#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define ENABLE_NOTE_LOG 0

#define CHECK(call)                                     \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

__global__ void flash_attn_1_fwd_f32_kernel(
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
{ // 1D-Block and 2D-Grid
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index in the grid

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y + by) * N * d; // gridDim.y = nh  qkv dim: (B, nh, N, d)   it equals to (by * gridDim.x + bx) * N * d
    int lm_offset = (bx * gridDim.y + by) * N;      // offset for l and m  lm dim: (B, nh, N)

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj , Bc >= Br, so choose consistent Bc as tile_size
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++)
    {

// Load Kj, Vj to SRAM
#pragma unroll
        for (int x = 0; x < d; x++) // one block caculates one tile
        {                           // tx * d indexes the thread, x indexes the element in the vector
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads(); // such that the inner loop can use the correct Kj, Vj

#pragma unroll
        for (int i = 0; i < Tr; i++)
        {

// Load Qi to SRAM, l and m to registers
#pragma unroll
            for (int x = 0; x < d; x++)
            { // one thread caculates one row of Qi
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            __syncthreads(); // such that the inner loop can use the correct Qi

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S) row-wise
            float row_m = -INFINITY;
#pragma unroll
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
#pragma unroll
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x]; // a thread caculates one element of S
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum; // S dim: (Br, Bc)

                if (sum > row_m)
                    row_m = sum; // every thread hold one row_m
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            // printf("%f", __expf(1.0));
#pragma unroll
            for (int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

// Write O, l, m to HBM
#pragma unroll
            for (int x = 0; x < d; x++)
            {
                float pv = 0; // Pij * Vj
#pragma unroll
                for (int y = 0; y < Bc; y++)
                {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

__global__ void
flash_attn_2_fwd_f32_kernel(
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

void initialize_random_array(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = (float)rand();
    }
}

int main(int argc, char const *argv[])
{
    float *Q, *K, *V, *l, *m, *O;

    Q = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    K = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    V = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    l = (float *)malloc(sizeof(float) * 16 * 71 * 128);
    m = (float *)malloc(sizeof(float) * 16 * 71 * 128);
    O = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);

    srand(time(nullptr));
    initialize_random_array(Q, 16 * 71 * 128 * 64);
    initialize_random_array(K, 16 * 71 * 128 * 64);
    initialize_random_array(V, 16 * 71 * 128 * 64);
    initialize_random_array(l, 16 * 71 * 128);
    initialize_random_array(m, 16 * 71 * 128);
    initialize_random_array(O, 16 * 71 * 128 * 64);

    const int B = 16;
    const int nh = 71;
    const int N = 128;
    const int d = 64;

    // to device
    float *d_Q_2, *d_K_2, *d_V_2, *d_l_2, *d_m_2, *d_O_2;
    CHECK(cudaMalloc((float **)&d_Q_2, sizeof(float) * 16 * 71 * 128 * 64));
    CHECK(cudaMalloc((float **)&d_K_2, sizeof(float) * 16 * 71 * 128 * 64));
    CHECK(cudaMalloc((float **)&d_V_2, sizeof(float) * 16 * 71 * 128 * 64));
    CHECK(cudaMalloc((float **)&d_l_2, sizeof(float) * 16 * 71 * 128));
    CHECK(cudaMalloc((float **)&d_m_2, sizeof(float) * 16 * 71 * 128));
    CHECK(cudaMalloc((float **)&d_O_2, sizeof(float) * 16 * 71 * 128 * 64));

    CHECK(cudaMemcpy(d_Q_2, Q, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K_2, K, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V_2, V, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l_2, l, sizeof(float) * 16 * 71 * 128, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m_2, m, sizeof(float) * 16 * 71 * 128, cudaMemcpyHostToDevice));

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
        d_Q_2, d_K_2, d_V_2, N, d, Tc, Tr, Bc, Br, softmax_scale, d_l_2, d_m_2, d_O_2);
    printf("Kernel_v2 done\n");

    float *d_Q_1, *d_K_1, *d_V_1, *d_l_1, *d_m_1, *d_O_1;
    CHECK(cudaMalloc((float **)&d_Q_1, sizeof(float) * 16 * 71 * 128 * 64));
    CHECK(cudaMalloc((float **)&d_K_1, sizeof(float) * 16 * 71 * 128 * 64));
    CHECK(cudaMalloc((float **)&d_V_1, sizeof(float) * 16 * 71 * 128 * 64));
    CHECK(cudaMalloc((float **)&d_l_1, sizeof(float) * 16 * 71 * 128));
    CHECK(cudaMalloc((float **)&d_m_1, sizeof(float) * 16 * 71 * 128));
    CHECK(cudaMalloc((float **)&d_O_1, sizeof(float) * 16 * 71 * 128 * 64));

    CHECK(cudaMemcpy(d_Q_1, Q, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K_1, K, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V_1, V, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_l_1, l, sizeof(float) * 16 * 71 * 128, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m_1, m, sizeof(float) * 16 * 71 * 128, cudaMemcpyHostToDevice));

    flash_attn_1_fwd_f32_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q_1, d_K_1, d_V_1, N, d, Tc, Tr, Bc, Br, softmax_scale, d_l_1, d_m_1, d_O_1);
    printf("Kernel_v1 done\n");

    // judge the kernel correctness between v1 and v2
    float *O_v1 = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    float *O_v2 = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    CHECK(cudaMemcpy(O_v1, d_O_1, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(O_v2, d_O_2, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 16 * 71 * 128 * 64; i++)
    {
        if (O_v1[i] != O_v2[i])
        {
            printf("Kernel_v1 and Kernel_v2 are not equal\n");
            break;
        }
    }
    cudaFree(d_Q_1);
    cudaFree(d_K_1);
    cudaFree(d_V_1);
    cudaFree(d_l_1);
    cudaFree(d_m_1);
    cudaFree(d_O_1);
    cudaFree(d_Q_2);
    cudaFree(d_K_2);
    cudaFree(d_V_2);
    cudaFree(d_l_2);
    cudaFree(d_m_2);
    cudaFree(d_O_2);
    cudaFreeHost(Q);
    cudaFreeHost(K);
    cudaFreeHost(V);
    cudaFreeHost(l);
    cudaFreeHost(m);
    cudaFreeHost(O);
    return 0;
}