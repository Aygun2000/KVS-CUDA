#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include "kv_store.h"
#include <cstdio>

void run_tests() {
    const int n = 100000;

    uint32_t* keys = new uint32_t[n];
    uint32_t* values = new uint32_t[n];
    uint32_t* query_keys = new uint32_t[n];
    uint32_t* results = new uint32_t[n];

    for (int i = 0; i < n; ++i) {
        keys[i] = i;
        values[i] = i * 10;
        query_keys[i] = i; 
    }

    uint32_t* d_keys, * d_values, * d_query_keys, * d_results;
    uint32_t* hash_table_keys, * hash_table_values;

    cudaMalloc(&d_keys, n * sizeof(uint32_t));
    cudaMalloc(&d_values, n * sizeof(uint32_t));
    cudaMalloc(&d_query_keys, n * sizeof(uint32_t));
    cudaMalloc(&d_results, n * sizeof(uint32_t));
    cudaMalloc(&hash_table_keys, TABLE_SIZE * sizeof(uint32_t));
    cudaMalloc(&hash_table_values, TABLE_SIZE * sizeof(uint32_t));

    cudaMemcpy(d_keys, keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_keys, query_keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(hash_table_keys, UINT32_MAX, TABLE_SIZE * sizeof(uint32_t));
    cudaMemset(hash_table_values, 0, TABLE_SIZE * sizeof(uint32_t));

    // GPU Timing Events for Insert
    cudaEvent_t insertStart, insertStop, lookupStart, lookupStop;
    cudaEventCreate(&insertStart);
    cudaEventCreate(&insertStop);
    cudaEventCreate(&lookupStart);
    cudaEventCreate(&lookupStop);
    float elapsed_time;

    // **GPU Insert Timing**
    cudaEventRecord(insertStart);
    kv_insert << <(n + 255) / 256, 256 >> > (d_keys, d_values, hash_table_keys, hash_table_values, n);
    cudaError_t err = cudaGetLastError();  
    if (err != cudaSuccess) {
        printf("CUDA Error (Insert Launch): %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(insertStop);
    cudaEventSynchronize(insertStop);
    cudaEventElapsedTime(&elapsed_time, insertStart, insertStop);
    printf("GPU Insert Time: %f ms\n", elapsed_time);

    // **GPU Lookup Timing**
    cudaEventRecord(lookupStart);
    kv_lookup << <(n + 255) / 256, 256 >> > (d_query_keys, d_results, hash_table_keys, hash_table_values, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (Lookup Launch): %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(lookupStop);
    cudaEventSynchronize(lookupStop);
    cudaEventElapsedTime(&elapsed_time, lookupStart, lookupStop);
    printf("GPU Lookup Time: %f ms\n", elapsed_time);

    // Free CUDA events
    cudaEventDestroy(insertStart);
    cudaEventDestroy(insertStop);
    cudaEventDestroy(lookupStart);
    cudaEventDestroy(lookupStop);

    // Free memory
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_query_keys);
    cudaFree(d_results);
    cudaFree(hash_table_keys);
    cudaFree(hash_table_values);

    delete[] keys;
    delete[] values;
    delete[] query_keys;
    delete[] results;
}

#ifdef GPU_TEST
int main() {
    printf("Starting GPU Key-Value Store Test...\n");
    run_tests();
    printf("GPU Test Completed.\n");
    return 0;
}
#endif
