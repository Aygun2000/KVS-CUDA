KVS-CUDA: Key-Value Store Using CUDA
KVS-CUDA is a high-performance key-value store designed for efficient key insertion and lookup using NVIDIA's CUDA technology. This project demonstrates how leveraging GPU parallelism can significantly accelerate database operations compared to traditional CPU-based approaches.

Overview
In large-scale databases, fast key-value retrieval is essential. Consider a scenario where a company manages a database of 1,000,000 game keys. When a user purchases a key, it must be quickly assigned and verified. Traditionally, this process relied on CPU computation, which could become a bottleneck for large datasets.

By utilizing CUDA and NVIDIA GPUs, we can achieve much faster key lookups and insertions, taking advantage of the superior parallel processing power of GPUs.

A detailed presentation is available in the repository.

Performance Benchmarking
CPU-Based Key-Value Store

Command:
.\cpu_kv_store.exe

Results:
📝 Insertion Time: 0.37489 sec
🔍 Lookup Time: 0.161414 sec
GPU-Based Key-Value Store

Command:
.\gpu_kv_store.exe

Results:
🚀 Insertion Time: 0.594944 ms
⚡ Lookup Time: 0.025600 ms
Extended Testing with 10,000 Keys
Method	Insertion Time	Lookup Time
CPU	0.387611 sec	0.140861 sec
GPU	0.513024 ms	0.076800 ms

Key Takeaways
✅ GPUs drastically outperform CPUs in key-value store operations.
✅ Lookup times are significantly reduced, making GPU acceleration ideal for large-scale databases.
✅ This approach can be applied to various real-world scenarios, such as game key management, caching systems, and high-performance computing.

Getting Started
Requirements
NVIDIA GPU with CUDA support
CUDA Toolkit installed
C++ compiler supporting CUDA
Building & Running
Clone the repository:
git clone https://github.com/your-repo/kvs-cuda.git

cd kvs-cuda
Compile the CPU version:
g++ cpu_kv_store.cpp -o cpu_kv_store.exe

Compile the GPU version:
nvcc gpu_kv_store.cu -o gpu_kv_store.exe

Run the tests:
./cpu_kv_store.exe  
./gpu_kv_store.exe  

Contributing
Contributions are welcome! Feel free to submit issues or pull requests to enhance performance or add new features.

License
This project is licensed under the MIT License.

