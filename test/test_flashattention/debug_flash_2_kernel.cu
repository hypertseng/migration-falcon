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
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                              (__expf(row_m - row_m_new) * row_l);

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
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1 / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) *
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
                     (__expf(row_m - row_m_new) * pv));
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
    float *L,
    float *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for L

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];

    for (int i = 0; i < Tr; ++i)
    {
        if (i * Br + tx >= N)
            break; // break if we are done with the sequence

        // Load Qi from HBM to SRAM, l and m to registers
        for (int x = 0; x < d; x++)
        {
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
        }
        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        // Causal mask: j <= i
        for (int j = 0; j <= i; ++j)
        {
            __syncthreads();
            // Load Kj, Vj from HBM to SRAM
            for (int x = 0; x < d; x++)
            {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads();
            // S_i^j = softmax_scale * QiKj^T
            // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++)
            {
                if (j * Bc + y >= N)
                    break; // break if we are done with the sequence
                if (i * Br + tx < j * Bc + y)
                    break;
                float sum = 0;
                for (int x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // m_i^j = max(m_i^j-1, row_max(S_i^j))
            float new_row_m = max(row_m_prev, row_m);

            // P_i^j = exp(S_i^j - m_i^j)
            // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
            float row_l = 0;
            for (int y = 0; y < Bc; y++)
            {
                if (j * Bc + y >= N)
                    break; // break if we are done with the sequence
                if (i * Br + tx < j * Bc + y)
                    break;
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - new_row_m);
                row_l += S[(Bc * tx) + y];
            }

            // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
            float row_m_exp = __expf(row_m_prev - new_row_m);
            float new_row_l = (row_m_exp * row_l_prev) + row_l;

            // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
            for (int x = 0; x < d; x++)
            {
                float pv = 0; // Pij * Vj
                for (int y = 0; y < Bc; y++)
                {
                    if (j * Bc + y >= N)
                        break; // break if we are done with the sequence
                    if (i * Br + tx < j * Bc + y)
                        break;
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
            }

            // Update m and l
            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
        }

        // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
        for (int x = 0; x < d; x++)
            O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
        // L_i = m_i^{Tc} + log(l_i^{Tc})
        L[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
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

        for (int i = j; i < Tr; i++)
        {
            __syncthreads();
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
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;
            }

            // Pij = diag(li)^-1 * exp(Sij - mi)
            // Pij[tx][y] = (1 / li[tx]) * exp(Sij[tx][y] - mi[tx])
            for (int y = 0; y < Bc; y++)
            {
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = (1 / l_curr) * __expf(S[(Bc * tx) + y] - m_curr);
            }
            __syncthreads();
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
            __syncthreads();
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

        // Upload Kj, Vj to HRAM
        for (int x = 0; x < d; x++)
        {
            dK[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dKj[(tx * d) + x];
            dV[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dVj[(tx * d) + x];
        }
    }
}

__global__ void flash_attn_2_bwd_f32_kernel(
    const float *Q,
    const float *K,
    const float *V,
    const float *O,
    const float *dO,
    const float *L,
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

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for L

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

        for (int i = j; i < Tr; i++)
        {
            __syncthreads();
            // Load Qi, Oi, dOi, dQi, li, mi to SRAM
            // Also load l, m to registers
            float Di = 0;
            for (int x = 0; x < d; x++)
            {
                Qi[(tx * d) + x] = Q[qkv_offset + (row_tile_size * i) + (tx * d) + x];
                Oi[(tx * d) + x] = O[qkv_offset + (row_tile_size * i) + (tx * d) + x];
                dOi[(tx * d) + x] = dO[qkv_offset + (row_tile_size * i) + (tx * d) + x];
                Di += dOi[(tx * d) + x] * Oi[(tx * d) + x];
            }
            float l_curr = L[lm_offset + (Br * i) + tx];

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
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;
            }

            // Pij = diag(li)^-1 * exp(Sij - mi)
            // Pij[tx][y] = (1 / li[tx]) * exp(Sij[tx][y] - mi[tx])
            for (int y = 0; y < Bc; y++)
            {
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - l_curr);
            }
            __syncthreads();
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
            __syncthreads();
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

        // Upload Kj, Vj to HRAM
        for (int x = 0; x < d; x++)
        {
            dK[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dKj[(tx * d) + x];
            dV[qkv_offset + (row_tile_size * j) + (tx * d) + x] = dVj[(tx * d) + x];
        }
    }
}

__host__ void initialize_random_array(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (time(nullptr) % 2)
            arr[i] = ((float)(rand() % 100)) / 100.0;
        else
            arr[i] = -((float)(rand() % 100)) / 100.0;
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
    // memset(Q, 0, sizeof(float) * 16 * 71 * 128 * 64);
    // memset(K, 0, sizeof(float) * 16 * 71 * 128 * 64);
    // memset(V, 0, sizeof(float) * 16 * 71 * 128 * 64);
    memset(l, 0, sizeof(float) * 16 * 71 * 128);
    for (int i = 0; i < 16 * 71 * 128; i++)
    {
        m[i] = -INFINITY;
    }
    memset(O, 0, sizeof(float) * 16 * 71 * 128 * 64);

    const int B = 16;
    const int nh = 71;
    const int N = 128;
    const int d = 64;

    // to device
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
    CHECK(cudaMemcpy(d_O_1, O, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice)); // O is initialized to 0

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
    CHECK(cudaMemcpy(d_O_2, O, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyHostToDevice));

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

    flash_attn_1_fwd_f32_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q_1, d_K_1, d_V_1, N, d, Tc, Tr, Bc, Br, softmax_scale, d_l_1, d_m_1, d_O_1);
    printf("Kernel_v1 done\n");

    flash_attn_2_fwd_f32_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q_2, d_K_2, d_V_2, N, d, Tc, Tr, Bc, Br, softmax_scale, d_l_2, d_O_2);
    printf("Kernel_v2 done\n");

    // judge the kernel correctness between v1 and v2
    float *O_v1 = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    float *O_v2 = (float *)malloc(sizeof(float) * 16 * 71 * 128 * 64);
    CHECK(cudaMemcpy(O_v1, d_O_1, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(O_v2, d_O_2, sizeof(float) * 16 * 71 * 128 * 64, cudaMemcpyDeviceToHost));
    // calculate the max diff
    float max_diff = 0;
    for (int i = 0; i < 16 * 71 * 128 * 64; i++)
    {
        float diff = abs(O_v1[i] - O_v2[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    printf("max_diff: %f\n", max_diff);
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
    cudaFreeHost(O_v1);
    cudaFreeHost(O_v2);
    return 0;
}