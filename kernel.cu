#include <stdio.h>
#include <cuda_runtime.h>
#include "kv_store.h"

int main() {
    const int n = 5;
    uint32_t keys[n] = { 1, 2, 3, 4, 5 };
    uint32_t values[n] = { 10, 20, 30, 40, 50 };
    uint32_t query_keys[n] = { 3, 6, 1, 4, 2 };
    uint32_t results[n] = { 0 };

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
    cudaMemset(hash_table_keys, 0xFF, TABLE_SIZE * sizeof(uint32_t)); 
    cudaMemset(hash_table_values, 0, TABLE_SIZE * sizeof(uint32_t));

    kv_insert << <1, n >> > (d_keys, d_values, hash_table_keys, hash_table_values, n);
    cudaDeviceSynchronize();

    kv_lookup << <1, n >> > (d_query_keys, d_results, hash_table_keys, hash_table_values, n);
    cudaDeviceSynchronize();

    cudaMemcpy(results, d_results, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Query results:\n");
    for (int i = 0; i < n; ++i) {
        if (results[i] == UINT32_MAX) {
            printf("Key: %d, Value: Not Found\n", query_keys[i]);
        }
        else {
            printf("Key: %d, Value: %d\n", query_keys[i], results[i]);
        }
    }

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_query_keys);
    cudaFree(d_results);
    cudaFree(hash_table_keys);
    cudaFree(hash_table_values);

    return 0;
}
