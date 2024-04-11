// Modified from: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ENABLE_NOTE_LOG 0

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

__global__ void flash_attn_1_bwd_f32_kernel(
    const float *Q,
    const float *K,
    const float *V,
    const float *O,
    const float *dO,
    const float *l,
    const float *m,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float *dQ,
    float *dK,
    float *dV)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int col_tile_size = Bc * d; // size of Kj, Vj
    int row_tile_size = Br * d; // size of Qi
    float *Kj = sram;
    float *Vj = &sram[col_tile_size];

    float *dKj = &sram[col_tile_size * 2];
    float *dVj = &sram[col_tile_size * 3];

    float *Qi = &sram[col_tile_size * 4];
    float *Oi = &sram[col_tile_size * 4 + row_tile_size];
    float *dOi = &sram[col_tile_size * 4 + row_tile_size * 2];

    // We also use S for P. Likewise, we use dS for dP.
    // We can reuse the same memory because we don't need S and P at the same time.
    // We also don't need dS and dP at the same time.
    float *S = &sram[col_tile_size * 4 + row_tile_size * 3];
    float *dS = &sram[col_tile_size * 4 + row_tile_size * 3 + Bc * Br];

    for (int j = 0; j < Tc; j++)
    {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++)
        {
            Kj[(tx * d) + x] = K[qkv_offset + (col_tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (col_tile_size * j) + (tx * d) + x];
        }

        // Initialize dKj, dVj to 0
        for (int x = 0; x < d; x++)
        {
            dKj[(tx * d) + x] = 0;
            dVj[(tx * d) + x] = 0;
        }

        __syncthreads(); // such that the inner loop can use the correct Kj, Vj
        for (int i = 0; i < Tr; i++)
        {

            // Load Qi, Oi, dOi, dQi, li, mi to SRAM
            // Also load l, m to registers
            for (int x = 0; x < d; x++)
            {
                Qi[(tx * d) + x] = Q[qkv_offset + (row_tile_size * i) + (tx * d) + x];
                Oi[(tx * d) + x] = O[qkv_offset + (row_tile_size * i) + (tx * d) + x];
                dOi[(tx * d) + x] = dO[qkv_offset + (row_tile_size * i) + (tx * d) + x];
            }
            float m_curr = m[lm_offset + (Br * i) + tx];
            float l_curr = l[lm_offset + (Br * i) + tx];

            // Sij = softmax_scale * QiKj^T
            // Sij[tx][y] = softmax_scale * Sum_{y = 0}^{Bc-1} Qi[tx][x] * Kj[y][x]
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
            }

            // Pij = diag(li)^-1 * exp(Sij - mi)
            // Pij[tx][y] = (1 / li[tx]) * exp(Sij[tx][y] - mi[tx])
            for (int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = (1 / l_curr) * __expf(S[(Bc * tx) + y] - m_curr);
            }

            // dVj <- dVj + Pij^T * dOi
            // dVj[tx][x] = dVj[tx][x] + Sum_{y = 0}^{Br-1} Pij[y][tx] * dOi[tx][x]
            for (int x = 0; x < d; x++)
            {
                float sum = 0;
                for (int y = 0; y < Br; y++)
                {
                    sum += S[(Bc * y) + tx] * dOi[(tx * d) + x];
                }
                atomicAdd(&dVj[(tx * d) + x], sum);
            }

            // dPij <- dOi * Vj^T
            // dPij[tx][y] = Sum_{x = 0}^{d-1} dOi[tx][x] * Vj[y][x]
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += dOi[(tx * d) + x] * Vj[(y * d) + x];
                }
                dS[(Bc * tx) + y] = sum;
            }

            // Di <- rowsum(dOi * Oi)  (pointwise multiply)
            // Di[tx] = Sum_{x = 0}^{d-1} dOi[tx][x] * Oi[tx][x]
            float Di = 0;
            for (int x = 0; x < d; x++)
            {
                Di += dOi[(tx * d) + x] * Oi[(tx * d) + x];
            }

            // dSij <- Pij * (dPij - Di)
            // dSij[tx][y] = Pij[tx][y] * (dPij[tx][y] - Di[tx])
            for (int y = 0; y < Bc; ++y)
            {
                dS[(Bc * tx) + y] = S[(Bc * tx) + y] * (dS[(Bc * tx) + y] - Di);
            }

            // dQi <- dQi + softmax_scale * dSijKj
            // dQ[tx][x] = dQ[tx][x] + softmax_scale * Sum_{y = 0}^{Bc-1} dSij[tx][y] * Kj[y][x]
            for (int x = 0; x < d; x++)
            {
                float sum = 0;
                for (int y = 0; y < Bc; y++)
                {
                    sum += dS[(Bc * tx) + y] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                atomicAdd(&dQ[qkv_offset + (row_tile_size * i) + (tx * d) + x], sum);
            }

            // dKj <- dKj + softmax_scale * dSij^TQi
            // dKj[tx][x] = dKj[tx][x] + softmax_scale * Sum_{y = 0}^{Br-1} dSij[y][tx] * Qi[y][x]
            for (int x = 0; x < d; x++)
            {
                float sum = 0;
                for (int y = 0; y < Br; y++)
                {
                    sum += dS[(Bc * y) + tx] * Qi[(y * d) + x];
                }
                sum *= softmax_scale;
                atomicAdd(&dKj[(tx * d) + x], sum);
            }
        }
        __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop

        // Upload Kj, Vj to HRAM
        for (int x = 0; x < d; x++)
        {
            dK[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dKj[(tx * d) + x];
            dV[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dVj[(tx * d) + x];
        }

        __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

extern "C"
{

    int flash_attn_1_fwd_f32(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra)
    {
        cudaStream_t custream = static_cast<cudaStream_t>(stream);
        if (nparam != 6)
            return 1;
        float *Q = static_cast<float *>(params[0]);
        float *K = static_cast<float *>(params[1]);
        float *V = static_cast<float *>(params[2]);
        // put l,m as input params
        float *l = static_cast<float *>(params[3]);
        float *m = static_cast<float *>(params[4]);
        float *O = static_cast<float *>(params[5]);

        const int B = static_cast<int>(shapes[0][0]);
        const int nh = static_cast<int>(shapes[0][1]);
        const int N = static_cast<int>(shapes[0][2]);
        const int d = static_cast<int>(shapes[0][3]);

        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

        // set block size, TODO: dynamically set block size
        const int Bc = 32;
        const int Br = 32;
        // const int Bc = ceil(max_sram_size / (4 * d * sizeof(float)));
        // const int Br = min(Bc, d);

        const int Tc = ceil((float)N / Bc);
        const int Tr = ceil((float)N / Br);
        const float softmax_scale = 1.0 / sqrt(d);

        // Calculate SRAM size needed per block
        const int sram_size = (2 * Bc * d * sizeof(float)) + (4 * Br * d * sizeof(float));
        printf("Bc: %d, Br: %d, Tc: %d, Tr: %d \n", Bc, Br, Tc, Tr);
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

        dim3 grid_dim(B, nh); // batch_size x num_heads
        dim3 block_dim(Bc);   // Bc threads per block

        flash_attn_1_fwd_f32_kernel<<<grid_dim, block_dim, sram_size, custream>>>(
            Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O);
        return 0;
    }
}

extern "C"
{

    int flash_attn_2_fwd_f32(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra)
    {
        cudaStream_t custream = static_cast<cudaStream_t>(stream);
        if (nparam != 6)
            return 1;
        float *Q = static_cast<float *>(params[0]);
        float *K = static_cast<float *>(params[1]);
        float *V = static_cast<float *>(params[2]);
        // put l,m as input params
        float *l = static_cast<float *>(params[3]);
        float *m = static_cast<float *>(params[4]);
        float *O = static_cast<float *>(params[5]);

        const int B = static_cast<int>(shapes[0][0]);
        const int nh = static_cast<int>(shapes[0][1]);
        const int N = static_cast<int>(shapes[0][2]);
        const int d = static_cast<int>(shapes[0][3]);

        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

        // set block size, TODO: dynamically set block size
        const int Bc = 32;
        const int Br = 32;

        const int Tc = ceil((float)N / Bc);
        const int Tr = ceil((float)N / Br);
        const float softmax_scale = 1.0 / sqrt(d);

        // Calculate SRAM size needed per block
        const int sram_size = (2 * Bc * d * sizeof(float)) + (4 * Br * d * sizeof(float));
        printf("Bc: %d, Br: %d, Tc: %d, Tr: %d \n", Bc, Br, Tc, Tr);
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

        dim3 grid_dim(B, nh); // batch_size x num_heads
        dim3 block_dim(Bc);   // Bc threads per block

        flash_attn_2_fwd_f32_kernel<<<grid_dim, block_dim, sram_size, custream>>>(
            Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O);
        return 0;
    }
}

// TODO: Implement backward pass