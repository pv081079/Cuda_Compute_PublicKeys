#include <stdio.h>
#include <stdint.h>
#include <curand_kernel.h>
#include <chrono>
#include "secp256k1/inc_vendor.h"
#include "secp256k1/inc_types.h"
#include "secp256k1/inc_platform.h"
#include "secp256k1/inc_common.h"
#include "secp256k1/inc_ecc_secp256k1.h"

__constant__ secp256k1_t s_basepoint;
unsigned int BLOCK_THREADS = 512; // Maximize the threads per block
unsigned int BLOCK_NUMBER = 0; // Set based on GPU properties dynamically

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        cudaDeviceReset(); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void init_curand_kernel(curandState* state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

// Device function to compute the compressed public key
__device__ void private_to_public(const uint32_t* pri, uint32_t* pub) {
    uint32_t a[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        a[i] = hc_swap32_S(pri[7 - i]);

    point_mul(pub, a, &s_basepoint);

    #pragma unroll
    for (int i = 0; i < 9; i++)
        pub[i] = hc_swap32_S(pub[i]);
}

// Device function to compute the uncompressed public key
__device__ void private_to_public_full(const uint32_t* pri, uint32_t* pub) {
    uint32_t a[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        a[i] = hc_swap32_S(pri[7 - i]);

    uint32_t x[8], y[8];
    point_mul_xy(x, y, a, &s_basepoint);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        pub[i] = hc_swap32_S(x[7 - i]);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        pub[i + 8] = hc_swap32_S(y[7 - i]);
}

// Kernel to generate a private key and compute both compressed and uncompressed public keys
__global__ void generate_keypair_kernel(curandState* state, uint32_t* prvKeys, uint32_t* compressedPubKeys, uint32_t* uncompressedPubKeys) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = state[id];

    // Generate a random private key
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        prvKeys[id * 8 + i] = curand(&localState);
    }

    // Compute the compressed public key
    private_to_public(&prvKeys[id * 8], &compressedPubKeys[id * 9]);

    // Compute the uncompressed public key
    private_to_public_full(&prvKeys[id * 8], &uncompressedPubKeys[id * 16]);

    state[id] = localState;
}

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    if (BLOCK_NUMBER == 0) {
        BLOCK_NUMBER = props.multiProcessorCount * 4;  // Set a high number of blocks to keep the GPU busy
    }

    fprintf(stderr, "[!] %s (%2d procs | Blocks: %d | Threads: %d)\n", props.name, props.multiProcessorCount, BLOCK_NUMBER, BLOCK_THREADS);

    secp256k1_t basepoint;
    set_precomputed_basepoint_g(&basepoint);
    cudaMemcpyToSymbol(s_basepoint, &basepoint, sizeof(basepoint));

    curandState* d_state;
    uint32_t* d_prvKeys;
    uint32_t* d_compressedPubKeys;
    uint32_t* d_uncompressedPubKeys;
    size_t totalThreads = BLOCK_NUMBER * BLOCK_THREADS;
    cudaMalloc((void**)&d_prvKeys, 8 * totalThreads * sizeof(uint32_t)); 
    cudaMalloc((void**)&d_compressedPubKeys, 9 * totalThreads * sizeof(uint32_t));
    cudaMalloc((void**)&d_uncompressedPubKeys, 16 * totalThreads * sizeof(uint32_t));
    cudaMalloc((void**)&d_state, totalThreads * sizeof(curandState));
    cudaCheckError();

    init_curand_kernel<<<BLOCK_NUMBER, BLOCK_THREADS>>>(d_state, time(0));
    cudaCheckError();

    // Variables for performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    unsigned long long totalKeysGenerated = 0;

    // Infinite loop to continuously generate keys and measure performance
    while (true) {
        // Launch the kernel to generate keys
        generate_keypair_kernel<<<BLOCK_NUMBER, BLOCK_THREADS>>>(d_state, d_prvKeys, d_compressedPubKeys, d_uncompressedPubKeys);
        cudaCheckError();

        totalKeysGenerated += totalThreads;

        // Measure time elapsed
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (elapsed.count() >= 1.0) {  // Output every second
            double keysPerSecond = totalKeysGenerated / elapsed.count();
            printf("Keys generated per second: %.2f\n", keysPerSecond);
            start = end;  // Reset the timer
            totalKeysGenerated = 0;  // Reset the counter
        }
    }

    // Cleanup (not reached due to infinite loop)
    cudaFree(d_prvKeys);
    cudaFree(d_compressedPubKeys);
    cudaFree(d_uncompressedPubKeys);
    cudaFree(d_state);

    return 0;
}
