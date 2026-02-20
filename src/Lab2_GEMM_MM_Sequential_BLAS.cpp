#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <Accelerate/Accelerate.h>

using namespace std;

/*
Author: Poorani T S
Assignment: Part 2: GEMM 
Date: 02/01/2026
*/

// Note: Below command works only on Mac systems with Accelerate framework. Installation BLAS is not required in this case.
// Instructions:
// 1. Compile the code using the command: clang++ -O0 -std=c++17 Lab2_GEMM_MM_Sequential_BLAS.cpp -framework Accelerate -o Lab2_GEMM_MM_Sequential_BLAS
// 2. Execute the code using the command: ./Lab2_GEMM_MM_Sequential_BLAS --range_start 10 --range_end 10000

// Using a type alias for cleaner code
using Matrix = vector<vector<double>>;

// Stats class to hold performance and error metrics for all methods
class Stats {
    public:
        int size;
        
        chrono::duration<double, nano> sequential_duration;
        chrono::duration<double, nano> blas_duration;
        
        double seq_vs_blas_avg_error;
        double seq_vs_blas_max_error;
        
        double seq_gflops;
        double blas_gflops;
    
    Stats(int s) {
        size = s;
    }

    // Compute GFLOPS for all methods using the formula: (2 * N^3) / duration_in_nanoseconds
    void compute_gflops() {
        double operations = 2.0 * size * size * size;
        seq_gflops = operations / sequential_duration.count();
        blas_gflops = operations / blas_duration.count();
    }

    // Function to print all stats
    void printStats() {
        cout << "----- Matrix Multiplication Stats -----\n";
        cout << "Matrix Size: " << size << "x" << size << "\n";
        cout << "---------------------------------------\n";
        cout << "Sequential Duration (ns): " << sequential_duration.count() << "\n";
        cout << "BLAS Duration (ns): " << blas_duration.count() << "\n";
        cout << fixed << setprecision(15);
        cout << "---------------------------------------\n";
        cout << "Seq vs BLAS Avg Error: " << seq_vs_blas_avg_error << "\n";
        cout << "---------------------------------------\n";
        cout << "Seq vs BLAS Max Error: " << seq_vs_blas_max_error << "\n";
        cout << "---------------------------------------\n";
        cout << fixed << setprecision(2);
        cout << "Sequential GFLOPS: " << seq_gflops << "\n";
        cout << "BLAS GFLOPS: " << blas_gflops << "\n";
        cout << "---------------------------------------\n\n\n";

    }
};

/** Method 1 - Sequential matrix-matrix multiplication */

Matrix performSequentialMatrixMultiplication(Matrix matrix1, Matrix matrix2, Stats& stats) {
    int size = matrix1.size();
    Matrix result(size, vector<double>(size, 0.0));
    
    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto seq_duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
    stats.sequential_duration = seq_duration;

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

Stats compareMatrixMultiplication(int size) {
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

    // Perform Sequential matrix-matrix multiplication
    Matrix sequential_result = performSequentialMatrixMultiplication(matrix1, matrix2, stats);

    // BLAS matrix-matrix multiplication using OpenBLAS
    Matrix blas_result = performBLASMatrixMultiplication(matrix1, matrix2, stats);

    // Compute and compare errors between results
    double seq_vs_blas_total_error = 0.0;
    double seq_vs_blas_max_error = 0.0;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double seq_val = sequential_result[i][j];
            double blas_val = blas_result[i][j];
            
            // Calculate errors
            double error_seq_blas = abs(seq_val - blas_val);

            // Accumulate total errors
            seq_vs_blas_total_error += error_seq_blas;

            // Update max errors
            if (error_seq_blas > seq_vs_blas_max_error) {
                seq_vs_blas_max_error = error_seq_blas;
            }
        }
    }

    int total_elements = size * size;
    
    // Calculate average errors
    stats.seq_vs_blas_avg_error = seq_vs_blas_total_error / total_elements;
    
    // Set max errors
    stats.seq_vs_blas_max_error = seq_vs_blas_max_error;

    // Compute GFLOPS and print stats
    stats.compute_gflops();
    stats.printStats();

    return stats;
};

// Application Configuration Structure
struct AppConfig {
    int range_start = 10;
    int range_end = 10000;

    // Helper to print current settings
    void print() const {
        std::cout << "--- Configuration ---\n"
                  << "  Range:   [" << range_start << ", " << range_end << "]\n"
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
    }
    return config;
};

// Function to write stats to CSV file
void writeStatsToCSV(const vector<Stats> all_stats, const AppConfig config) {
        string filename = "mm_multi_seq_stats_"+to_string(config.range_start)+"_"+to_string(config.range_end)+".csv";

    ofstream outfile(filename);
    outfile << "Size,Sequential Duration (ns),BLAS Duration (ns),"
               "Seq vs BLAS Avg Error,"
               "Seq vs BLAS Max Error,"
               "Seq GFLOPS,BLAS GFLOPS\n";   
    for (const auto& stats : all_stats) {
        outfile << stats.size << ","
                << stats.sequential_duration.count() << ","
                << stats.blas_duration.count() << ","
                << stats.seq_vs_blas_avg_error << ","
                << stats.seq_vs_blas_max_error << ","
                << stats.seq_gflops << ","
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
        Stats stats = compareMatrixMultiplication(size);
        all_stats.push_back(stats);

        if (size < 500) size += 10;
        else if (size < 1000) size += 20;
        else size += 1000;
    }

    // Write results to csv file with header and custom filename
    writeStatsToCSV(all_stats, config);
    
    return 0;
}
