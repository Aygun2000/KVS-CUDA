#include "kv_store.h"

__global__ void kv_insert(uint32_t* keys, uint32_t* values, uint32_t* hash_table_keys, uint32_t* hash_table_values, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {  
        int hash = keys[idx] % TABLE_SIZE;

        while (atomicCAS(&hash_table_keys[hash], UINT32_MAX, keys[idx]) != UINT32_MAX) {
            hash = (hash + 1) % TABLE_SIZE;
        }
        hash_table_values[hash] = values[idx];
    }
}

__global__ void kv_lookup(uint32_t* query_keys, uint32_t* results, uint32_t* hash_table_keys, uint32_t* hash_table_values, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) { 
        int hash = query_keys[idx] % TABLE_SIZE;

        while (hash_table_keys[hash] != UINT32_MAX && hash_table_keys[hash] != query_keys[idx]) {
            hash = (hash + 1) % TABLE_SIZE;
        }
        if (hash_table_keys[hash] == query_keys[idx]) {
            results[idx] = hash_table_values[hash];
        }
        else {
            results[idx] = UINT32_MAX;
        }
    }
}
