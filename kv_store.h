#ifndef KV_STORE_H
#define KV_STORE_H

#include <cuda_runtime.h>
#include <stdint.h>

#define TABLE_SIZE 1024 

__global__ void kv_insert(uint32_t* keys, uint32_t* values, uint32_t* hash_table_keys, uint32_t* hash_table_values, int n);
__global__ void kv_lookup(uint32_t* query_keys, uint32_t* results, uint32_t* hash_table_keys, uint32_t* hash_table_values, int n);

#endif 
