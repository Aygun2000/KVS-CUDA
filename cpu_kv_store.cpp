#include <iostream>
#include <unordered_map>
#include <chrono>

void cpu_kv_store(int* keys, int* values, int n, std::unordered_map<int, int>& kv_store) {
    for (int i = 0; i < n; i++) {
        kv_store[keys[i]] = values[i];
    }
}

void cpu_kv_lookup(int* query_keys, int* results, int n, std::unordered_map<int, int>& kv_store) {
    for (int i = 0; i < n; i++) {
        auto it = kv_store.find(query_keys[i]);
        results[i] = (it != kv_store.end()) ? it->second : -1; // -1 for not found
    }
}
#ifdef CPU_TEST
int main() {
    const int n = 1000000;
    int* keys = new int[n];
    int* values = new int[n];
    int* query_keys = new int[n];
    int* results = new int[n];

    std::unordered_map<int, int> kv_store;

    for (int i = 0; i < n; i++) {
        keys[i] = i;
        values[i] = i * 10;
        query_keys[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cpu_kv_store(keys, values, n, kv_store);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Insert Time: " << std::chrono::duration<double>(end - start).count() << " sec\n";

    start = std::chrono::high_resolution_clock::now();
    cpu_kv_lookup(query_keys, results, n, kv_store);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Lookup Time: " << std::chrono::duration<double>(end - start).count() << " sec\n";

    delete[] keys;
    delete[] values;
    delete[] query_keys;
    delete[] results;
    return 0;
}
#endif