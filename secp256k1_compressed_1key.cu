#include <stdio.h>
#include <stdint.h>
#include <curand_kernel.h>
#include <chrono>
#include <cstring>  // Include for memcpy
#include "secp256k1/inc_vendor.h"
#include "secp256k1/inc_types.h"
#include "secp256k1/inc_platform.h"
#include "secp256k1/inc_common.h"
#include "secp256k1/inc_ecc_secp256k1.h"

__constant__ secp256k1_t s_basepoint;
unsigned int BLOCK_THREADS = 1;
unsigned int BLOCK_NUMBER = 1;

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

__device__ void private_to_public(const uint32_t* pri, uint32_t* pub) {
    uint32_t a[8];
    for (int i = 0; i < 8; i++)
        a[i] = hc_swap32_S(pri[7 - i]);

    point_mul(pub, a, &s_basepoint);

    for (int i = 0; i < 9; i++)
        pub[i] = hc_swap32_S(pub[i]);
}

__global__ void generate_keypair_kernel(curandState* state, uint32_t* prvKeys, uint32_t* compressedPubKeys) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = state[id];

    // Generate a random private key
    for (int i = 0; i < 8; i++) {
        prvKeys[i] = curand(&localState);
    }

    // Compute the public key from the private key (compressed form)
    private_to_public(prvKeys, compressedPubKeys);
    state[id] = localState;
}

// Helper function to reverse the byte order of a 32-bit integer
uint32_t reverse_bytes(uint32_t val) {
    return ((val >> 24) & 0x000000FF) |
           ((val >> 8)  & 0x0000FF00) |
           ((val << 8)  & 0x00FF0000) |
           ((val << 24) & 0xFF000000);
}

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    fprintf(stderr, "[!] %s (%2d procs | Blocks: %d | Threads: %d)\n", props.name, props.multiProcessorCount, BLOCK_NUMBER, BLOCK_THREADS);

    secp256k1_t basepoint;
    set_precomputed_basepoint_g(&basepoint);
    cudaMemcpyToSymbol(s_basepoint, &basepoint, sizeof(basepoint));

    curandState* d_state;
    uint32_t* d_prvKeys;
    uint32_t* d_compressedPubKeys;
    cudaMalloc((void**)&d_prvKeys, 8 * sizeof(uint32_t)); 
    cudaMalloc((void**)&d_compressedPubKeys, 9 * sizeof(uint32_t));
    cudaMalloc((void**)&d_state, sizeof(curandState));
    cudaCheckError();

    init_curand_kernel<<<BLOCK_NUMBER, BLOCK_THREADS>>>(d_state, time(0));
    cudaCheckError();

    // Run the key generation kernel once
    generate_keypair_kernel<<<BLOCK_NUMBER, BLOCK_THREADS>>>(d_state, d_prvKeys, d_compressedPubKeys);
    cudaCheckError();

    cudaDeviceSynchronize();  

    // Copy the generated keys back to host
    uint32_t h_prvKeys[8];
    uint32_t h_compressedPubKeys[9];
    cudaMemcpy(h_prvKeys, d_prvKeys, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_compressedPubKeys, d_compressedPubKeys, 9 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Print the private key with corrected byte order
    printf("Private Key: ");
    for (int i = 0; i < 8; i++) {
        uint32_t reversed = reverse_bytes(h_prvKeys[i]);
        printf("%02x%02x%02x%02x", 
            (reversed >> 24) & 0xFF, 
            (reversed >> 16) & 0xFF, 
            (reversed >> 8) & 0xFF, 
            reversed & 0xFF);
    }
    printf("\n");

    // Convert the public key to a uint8_t array with byte order correction
    uint8_t pub[33];
    memcpy(pub, (uint8_t*)h_compressedPubKeys, 33);

    // Print the public key (compressed) with corrected byte order
    printf("Compressed Public Key: ");
    for (int i = 0; i < 9; i++) {
        uint32_t reversed = reverse_bytes(h_compressedPubKeys[i]);
        for (int j = 0; j < 4; j++) {
            if (i * 4 + j < 33) {
                printf("%02x", (reversed >> (24 - j * 8)) & 0xFF);
            }
        }
    }
    printf("\n");

    // Cleanup
    cudaFree(d_prvKeys);
    cudaFree(d_compressedPubKeys);
    cudaFree(d_state);

    return 0;
}
