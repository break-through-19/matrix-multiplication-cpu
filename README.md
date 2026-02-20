# matrix-multiplication-cpu
This repository compares matrix multiplication of sequential, parallel and BLAS implementation for varying sizes of matrix.

## 1. Configurations and Devices Used

### Hardware Architectures
- **Apple M1**: Utilized unified memory architecture, achieving high throughput and stability.
- **Apple M4**: Demonstrated the most stable behavior in floating-point arithmetic results, likely due to an improved cache hierarchy and optimized library functions.
- **AMD Ryzen 7 5800H**: Showcased high sequential performance owing to high clock frequency.

### Matrix and Thread Configurations
- **Matrix Sizes (N)**: Ranging from 10 x 10 up to 10,000 x 10,000.
- **Thread Counts (T)**: Tested with 8, 10, and 16 threads using row decomposition strategy.

## 2. Highlights and Key Findings

- **BLAS Superiority**: The optimized BLAS implementation drastically outperformed naive sequential and multi-threaded approaches across all architectures. On the Apple M1 processor, BLAS achieved a maximum of **260 GFLOPS** using NEON vectorization and efficient cache tiling.
- **Multi-threading Bottlenecks**: 
  - **Cache Thrashing**: Significant performance degradation was observed as matrix size (N) increased due to cache thrashing. (Solution: Block Matrix Multiplication / Tiling).
  - **False Sharing**: Performance instability in threaded implementations at high N values was attributed to multiple threads modifying variables on the same cache line.
- **Cache Coherence**: The AMD Ryzen processor's multi-threaded performance was severely affected by L3 cache coherence issues compared to the Apple M1.
- **Floating-Point Precision**: Precision behavior varied. M1 showed volatility based on cache alignment, AMD showed monotonic increases, and M4 remained highly stable.

## 3. Tech Stack
- **Language**: C++ (Compiled with C++17 standard)
- **Compiler**: `clang++`
- **Parallelism**: Native `std::thread`
- **Optimized Libraries**: Apple Accelerate Framework (`cblas_dgemm`)

## 4. Significance of BLAS (Basic Linear Algebra Subprograms)

BLAS provides highly optimized routines for fundamental linear algebra operations. Its significance in this project lies in taking advantage of hardware-level optimizations without altering the high-level source code:
- **Auto-Vectorization**: BLAS automatically uses SIMD (Single Instruction, Multiple Data) instructions built into the hardware (like NEON on Apple Silicon or AVX on x86/AMD). This allows the program to process multiple data points simultaneously, running 10x to 50x faster.
- **Algorithmic & Memory Optimizations (Tiling & Sequencing)**: BLAS natively reorganizes matrices (transposing/tiling) to keep data "hot" in the cache. It prevents "strided" memory access penalties by ensuring sequential row-major reads, fully exploiting spatial locality and hardware prefetching mechanisms.
