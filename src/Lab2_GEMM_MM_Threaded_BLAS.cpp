#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <thread>
#include <Accelerate/Accelerate.h>

using namespace std;

/*
Author: Poorani T S
Assignment: Part 2: GEMM 
Date: 02/01/2026
*/

// Note: Below command works only on Mac systems with Accelerate framework. Installation BLAS is not required in this case.
// Instructions:
// 1. Compile the code using the command: clang++ -O0 -std=c++17 Lab2_GEMM_MM_Threaded_BLAS.cpp -framework Accelerate -o Lab2_GEMM_MM_Threaded_BLAS
// 2. Execute the code using the command: ./Lab2_GEMM_MM_Threaded_BLAS --range_start 10 --range_end 10000 --threads_count 8

// Using a type alias for cleaner code
using Matrix = vector<vector<double>>;

// Stats class to hold performance and error metrics for all methods
class Stats {
    public:
        int size;
        
        chrono::duration<double, nano> thread_duration;
        chrono::duration<double, nano> blas_duration;
        
        double thread_vs_blas_avg_error;
        double thread_vs_blas_max_error;
        
        double thread_gflops;
        double blas_gflops;
    
    Stats(int s) {
        size = s;
    }

    // Compute GFLOPS for all methods using the formula: (2 * N^3) / duration_in_nanoseconds
    void compute_gflops() {
        double operations = 2.0 * size * size * size;
        thread_gflops = operations / thread_duration.count();
        blas_gflops = operations / blas_duration.count();
    }

    // Function to print all stats
    void printStats() {
        cout << "----- Matrix Multiplication Stats -----\n";
        cout << "Matrix Size: " << size << "x" << size << "\n";
        cout << "---------------------------------------\n";
        cout << "Threaded Duration (ns): " << thread_duration.count() << "\n";
        cout << "BLAS Duration (ns): " << blas_duration.count() << "\n";
        cout << fixed << setprecision(15);
        cout << "---------------------------------------\n";
        cout << "Thread vs BLAS Avg Error: " << thread_vs_blas_avg_error << "\n";
        cout << "---------------------------------------\n";
        cout << "Thread vs BLAS Max Error: " << thread_vs_blas_max_error << "\n";
        cout << "---------------------------------------\n";
        cout << fixed << setprecision(2);
        cout << "Threaded GFLOPS: " << thread_gflops << "\n";
        cout << "BLAS GFLOPS: " << blas_gflops << "\n";
        cout << "---------------------------------------\n\n\n";

    }
};

/** Method 1 - Multi-threaded matrix-matrix multiplication */

/**
 * Thread Worker Function
 * * @param matA      Reference to input Matrix A (Read-only)
 * @param matB      Reference to input Matrix B (Read-only)
 * @param matC      Reference to result Matrix C (Write-only)
 * @param start_row The starting row index (inclusive) for this thread
 * @param end_row   The ending row index (exclusive) for this thread
 */
void multiply_worker(const Matrix matA, const Matrix matB, Matrix& matC, int start_row, int end_row, int size) {
    // Iterate only through the rows assigned to this specific thread
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < size; ++j) {
            matC[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                // Standard dot product: Row A[i] * Col B[j]
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
};

// Driver function for multi-threaded matrix multiplication
Matrix performThreadedMatrixMultiplication(Matrix matrix1, Matrix matrix2, int num_threads, Stats& stats) {
    int size = matrix1.size();
    Matrix result(size, vector<double>(size, 0.0));
    
    vector<thread> threads;
    int rows_per_thread = size / num_threads;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Launch threads to perform multiplication using Row Decomposition strategy
    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? size : start_row + rows_per_thread;
        
        threads.emplace_back(multiply_worker, matrix1, matrix2, ref(result), start_row, end_row, size);
    }
    
    // Join threads
    for (auto& th : threads) {
        th.join();
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto thread_duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
    stats.thread_duration = thread_duration;

    return result;
};


/** Method 2 - BLAS matrix-matrix multiplication using Apple Accelerate BLAS */
Matrix performBLASMatrixMultiplication(Matrix matrix1, Matrix matrix2, Stats& stats) {
    int size = matrix1.size();
    Matrix result(size, vector<double>(size, 0.0));
    
    // Flatten matrices for cblas_dgemm
    vector<double> flatA(size * size);
    vector<double> flatB(size * size);
    vector<double> flatC(size * size, 0.0);
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flatA[i * size + j] = matrix1[i][j];
            flatB[i * size + j] = matrix2[i][j];
        }
    }
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size,
                1.0,
                flatA.data(), size,
                flatB.data(), size,
                0.0,
                flatC.data(), size);
    
    auto end_time = chrono::high_resolution_clock::now();
    auto blas_duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
    stats.blas_duration = blas_duration;
    
    // Un-flatten result matrix
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i][j] = flatC[i * size + j];
        }
    }
    
    return result;
};

Stats compareMatrixMultiplication(int size, int num_threads) {
    // Declare matrix and matrix of given size
    Matrix matrix1(size, vector<double>(size));
    Matrix matrix2(size, vector<double>(size));

    // Initialize reproducible random number generator
    mt19937 gen(42);
    uniform_real_distribution<> dis(0.0, 100.0);

    // Fill matrix with random values
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix1[i][j] = dis(gen);
            matrix2[i][j] = dis(gen);
        }
    }
    Stats stats = Stats(size);

    // Perform Multi-threaded matrix-matrix multiplication
    Matrix threaded_result = performThreadedMatrixMultiplication(matrix1, matrix2, num_threads, stats);

    // BLAS matrix-matrix multiplication using OpenBLAS
    Matrix blas_result = performBLASMatrixMultiplication(matrix1, matrix2, stats);

    // Compute and compare errors between results
    double thread_vs_blas_total_error = 0.0;        
    double thread_vs_blas_max_error = 0.0;  

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double thread_val = threaded_result[i][j];
            double blas_val = blas_result[i][j];

            // Calculate errors
            double error_thread_blas = abs(thread_val - blas_val);

            // Accumulate total errors
            thread_vs_blas_total_error += error_thread_blas;

            // Update max errors
            if (error_thread_blas > thread_vs_blas_max_error) {
                thread_vs_blas_max_error = error_thread_blas;
            }
        }
    }

    // Calculate average errors
    int total_elements = size * size;
    stats.thread_vs_blas_avg_error = thread_vs_blas_total_error / total_elements;   

    // Set max errors
    stats.thread_vs_blas_max_error = thread_vs_blas_max_error;  

    // Compute GFLOPS and print stats
    stats.compute_gflops();
    stats.printStats();

    return stats;
};

// Application Configuration Structure
struct AppConfig {
    int range_start = 10;
    int range_end = 100;
    int threads_count = 8;

    // Helper to print current settings
    void print() const {
        std::cout << "--- Configuration ---\n"
                  << "  Range:   [" << range_start << ", " << range_end << "]\n"
                  << "  Threads: " << threads_count << "\n"
                  << "---------------------\n";
    }
};

// Function to parse command-line arguments
AppConfig parseArguments(int argc, char* argv[]) {
    AppConfig config; // Starts with defaults

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--range_start") {
            if (i + 1 < argc) config.range_start = std::stoi(argv[++i]);
            else std::cerr << "Warning: Missing value for --range_start, using default.\n";
        } 
        else if (arg == "--range_end") {
            if (i + 1 < argc) config.range_end = std::stoi(argv[++i]);
            else std::cerr << "Warning: Missing value for --range_end, using default.\n";
        } 
        else if (arg == "--threads_count") {
            if (i + 1 < argc) config.threads_count = std::stoi(argv[++i]);
            else std::cerr << "Warning: Missing value for --threads_count, using default.\n";
        }
    }
    return config;
};

// Function to write stats to CSV file
void writeStatsToCSV(const vector<Stats> all_stats, const AppConfig config) {
        string filename = "mm_multi_threaded_stats_"+to_string(config.range_start)+"_"+to_string(config.range_end)+"_t"+to_string(config.threads_count)+".csv";

    ofstream outfile(filename);
    outfile << "Size,Threaded Duration (ns),BLAS Duration (ns),"
               "Thread vs BLAS Avg Error,"
               "Thread vs BLAS Max Error,"
               "Thread GFLOPS,BLAS GFLOPS\n";   
    for (const auto& stats : all_stats) {
        outfile << stats.size << ","
                << stats.thread_duration.count() << ","
                << stats.blas_duration.count() << ","
                << stats.thread_vs_blas_avg_error << ","
                << stats.thread_vs_blas_max_error << ","
                << stats.thread_gflops << ","
                << stats.blas_gflops << "\n";
    }
    outfile.close();   
}

// Main function
int main(int argc, char* argv[]) {
    // Parse command-line arguments
    AppConfig config = parseArguments(argc, argv);

    // Validate logic "post-parsing"
    if (config.range_start >= config.range_end) {
        std::cerr << "Error: range_start must be less than range_end.\n";
        return 1;
    }

    vector<Stats> all_stats;
    int size = config.range_start;
    
    // Loop through different matrix sizes and collect stats
    while (size <= config.range_end) {
        Stats stats = compareMatrixMultiplication(size, config.threads_count);
        all_stats.push_back(stats);

        if (size < 500) size += 10;
        else if (size < 1000) size += 20;
        else size += 1000;
    }

    // Write results to csv file with header and custom filename
    writeStatsToCSV(all_stats, config);
    
    return 0;
}
