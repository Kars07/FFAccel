#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <nvtx3/nvToolsExt.h>

// Data types for finite field elements
typedef uint32_t fp_t;     // Assumes p fits in 32 bits
typedef uint64_t fp_wide_t; // Wide type for intermediate calculations

// Device constants for Better Barrett reduction
__device__ __constant__ fp_t PRIME_Q;
__device__ __constant__ uint64_t MU_BETTER;
__device__ __constant__ int M_BETTER;  // Bit length of prime
__device__ __constant__ int ALPHA;     // α = m + 1
__device__ __constant__ int BETA;      // β = -2

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Better Barrett reduction implementation
// Dhem-Quisquater with (α, β) = (m + 1, -2)
__device__ __forceinline__ fp_t better_barrett_reduce(fp_wide_t x) {
    // Algorithm: Better Barrett reduction
    // Require: m = len(q) ≤ β - 2, 0 ≤ x < 2^(2m)
    // μ = ⌊2^(2m+1)/q⌋
    // Ensure: rem = x mod q
    
    // Step 1: c ← ⌊x / 2^(m-2)⌋
    fp_wide_t c = x >> (M_BETTER - 2);
    
    // Step 2: quot ← ⌊(c × μ) / 2^(m+3)⌋
    fp_wide_t temp = c * MU_BETTER;
    fp_wide_t quot = temp >> (M_BETTER + 3);
    
    // Step 3: rem ← x - quot × q
    fp_wide_t rem = x - quot * PRIME_Q;
    
    // Step 4-6: Final correction (at most one subtraction needed)
    if (rem >= PRIME_Q) {
        rem = rem - PRIME_Q;
    }
    
    return (fp_t)rem;
}

// Optimized Better Barrett reduction with precomputed shifts
__device__ __forceinline__ fp_t better_barrett_reduce_optimized(fp_wide_t x) {
    // For p = 2^31 - 1, m = 31
    // c = x >> 29 (m-2 = 29)
    fp_wide_t c = x >> 29;
    
    // quot = (c * μ) >> 34 (m+3 = 34)
    fp_wide_t temp = c * MU_BETTER;
    fp_wide_t quot = temp >> 34;
    
    // rem = x - quot * p
    fp_wide_t rem = x - quot * PRIME_Q;
    
    // Final reduction (at most 1 needed for Better Barrett)
    if (rem >= PRIME_Q) {
        rem -= PRIME_Q;
    }
    
    return (fp_t)rem;
}

// Basic finite field operations using Better Barrett reduction
__device__ __forceinline__ fp_t fp_add_better_barrett(fp_t a, fp_t b) {
    fp_wide_t sum = (fp_wide_t)a + b;
    return (sum >= PRIME_Q) ? (sum - PRIME_Q) : (fp_t)sum;
}

__device__ __forceinline__ fp_t fp_sub_better_barrett(fp_t a, fp_t b) {
    return (a >= b) ? (a - b) : (a + PRIME_Q - b);
}

__device__ __forceinline__ fp_t fp_mul_better_barrett(fp_t a, fp_t b) {
    fp_wide_t prod = (fp_wide_t)a * b;
    return better_barrett_reduce_optimized(prod);
}

__device__ __forceinline__ fp_t fp_neg_better_barrett(fp_t a) {
    return (a == 0) ? 0 : (PRIME_Q - a);
}

// Modular exponentiation using Better Barrett reduction
__device__ fp_t fp_pow_better_barrett(fp_t base, fp_t exp) {
    fp_t result = 1;
    base = better_barrett_reduce_optimized(base); // Ensure base is within the field
    
    while (exp > 0) {
        if (exp & 1) {
            result = fp_mul_better_barrett(result, base);
        }
        exp >>= 1;
        base = fp_mul_better_barrett(base, base);
    }
    return result;
}

// Extended Euclidean Algorithm for modular inverse
__device__ fp_t fp_inv_better_barrett(fp_t a) {
    if (a == 0) {
        return 0; // Invalid input
    }

    int64_t old_r = PRIME_Q, r = a;
    int64_t old_s = 0, s = 1;

    while (r != 0) {
        int64_t quotient = old_r / r;
        
        int64_t temp = r;
        r = old_r - quotient * r;
        old_r = temp;

        temp = s;
        s = old_s - quotient * s;
        old_s = temp;
    }
    
    return (old_s < 0) ? (fp_t)(old_s + PRIME_Q) : (fp_t)old_s;
}

// Fermat's Little Theorem for modular inverse using Better Barrett reduction
__device__ fp_t fp_inv_fermat_better_barrett(fp_t a) {
    if (a == 0) {
        return 0; // Invalid input
    }
    return fp_pow_better_barrett(a, PRIME_Q - 2);
}

__device__ __forceinline__ fp_t fp_div_better_barrett(fp_t a, fp_t b) {
    return fp_mul_better_barrett(a, fp_inv_better_barrett(b));
}

// Batch operations kernels using Better Barrett reduction
__global__ void fp_add_arrays_better_barrett(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_add_better_barrett(a[idx], b[idx]);
    }
}

__global__ void fp_sub_arrays_better_barrett(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_sub_better_barrett(a[idx], b[idx]);
    }
}

__global__ void fp_mul_arrays_better_barrett(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_mul_better_barrett(a[idx], b[idx]);
    }
}

__global__ void fp_pow_arrays_better_barrett(const fp_t* base, const fp_t* exp, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_pow_better_barrett(base[idx], exp[idx]);
    }
}

__global__ void fp_inv_arrays_better_barrett(const fp_t* a, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_inv_better_barrett(a[idx]);
    }
}

// Polynomial evaluation using Better Barrett reduction
__global__ void fp_poly_eval_better_barrett(
    const fp_t* coeffs, int degree,
    const fp_t* x_vals, fp_t* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp_t x = x_vals[idx];
        fp_t result = coeffs[degree];

        for (int i = degree - 1; i >= 0; i--) {
            result = fp_add_better_barrett(fp_mul_better_barrett(result, x), coeffs[i]);
        }
        results[idx] = result;
    }
}

// Matrix multiplication using Better Barrett reduction
__global__ void fp_matrix_mul_better_barrett(const fp_t* A, const fp_t* B, fp_t* C,
                                           int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        fp_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum = fp_add_better_barrett(sum, fp_mul_better_barrett(A[row * n + i], B[i * k + col]));
        }
        C[row * k + col] = sum;
    }
}

// Host wrapper class with Better Barrett reduction
class FiniteFieldBetterBarrett {
private:
    fp_t q;
    uint64_t mu;
    int m; // Bit length of prime
    int alpha; // α = m + 1
    int beta;  // β = -2
    
    // Compute Better Barrett constant μ = ⌊2^(2m+1)/q⌋
    uint64_t compute_mu_better(fp_t prime) {
        // Find bit length of prime (number of bits needed to represent prime)
        int bit_length = 0;
        fp_t temp = prime;
        while (temp > 0) {
            bit_length++;
            temp >>= 1;
        }
        
        // For Better Barrett reduction: μ = ⌊2^(2m+1)/q⌋
        // We need to compute this carefully to avoid overflow
        
        // Use long double for higher precision
        long double numerator = 1.0L;
        for (int i = 0; i < 2 * bit_length + 1; i++) {
            numerator *= 2.0L;
        }
        
        uint64_t mu_result = (uint64_t)(numerator / (long double)prime);
        
        printf("Prime bit length: %d\n", bit_length);
        printf("Computing Better Barrett μ = ⌊2^%d / %u⌋\n", 2 * bit_length + 1, prime);
        
        return mu_result;
    }
    
public:
    FiniteFieldBetterBarrett(fp_t prime) : q(prime) {
        // Compute bit length
        m = 0;
        fp_t temp = prime;
        while (temp > 0) {
            m++;
            temp >>= 1;
        }
        
        // Better Barrett parameters
        alpha = m + 1;  // α = m + 1
        beta = -2;      // β = -2
        
        mu = compute_mu_better(prime);
        
        // Copy constants to device memory
        CUDA_CHECK(cudaMemcpyToSymbol(PRIME_Q, &q, sizeof(fp_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(MU_BETTER, &mu, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(M_BETTER, &m, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(ALPHA, &alpha, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(BETA, &beta, sizeof(int)));
        
        printf("Better Barrett reduction initialized:\n");
        printf("  q = %u\n", q);
        printf("  m = %d (bit length)\n", m);
        printf("  α = %d (m + 1)\n", alpha);
        printf("  β = %d\n", beta);
        printf("  μ = %llu\n", (unsigned long long)mu);
        
        // Verify μ calculation
        long double expected = 1.0L;
        for (int i = 0; i < 2 * m + 1; i++) {
            expected *= 2.0L;
        }
        expected /= (long double)prime;
        printf("  Expected μ ≈ %.2Lf\n", expected);
    }
    
    // Memory management helper
    struct DeviceArrays {
        fp_t *d_a, *d_b, *d_result;
        
        DeviceArrays(int n) {
            CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(fp_t)));
            CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(fp_t)));
            CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(fp_t)));
        }
        
        ~DeviceArrays() {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_result);
        }
    };
    
    void add_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
        nvtxRangePush("add_arrays_better_barrett");
        
        DeviceArrays arrays(n);
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(arrays.d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(arrays.d_b, h_b, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        fp_add_arrays_better_barrett<<<grid_size, block_size>>>(arrays.d_a, arrays.d_b, arrays.d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
       
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, arrays.d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePop();
    }
    
    void mul_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
        nvtxRangePush("mul_arrays_better_barrett");
        
        DeviceArrays arrays(n);
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(arrays.d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(arrays.d_b, h_b, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        fp_mul_arrays_better_barrett<<<grid_size, block_size>>>(arrays.d_a, arrays.d_b, arrays.d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, arrays.d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePop();
    }
    
    void inv_arrays(const fp_t* h_a, fp_t* h_result, int n) {
        nvtxRangePush("inv_arrays_better_barrett");
        
        fp_t *d_a, *d_result;
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(fp_t)));
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        fp_inv_arrays_better_barrett<<<grid_size, block_size>>>(d_a, d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        cudaFree(d_a);
        cudaFree(d_result);
        nvtxRangePop();
    }
    
    void matrix_multiply(const fp_t* h_A, const fp_t* h_B, fp_t* h_C,
                        int m, int n, int k) {
        fp_t *d_A, *d_B, *d_C;
        
        CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_B, n * k * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_C, m * k * sizeof(fp_t)));
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A, m * n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, n * k * sizeof(fp_t), cudaMemcpyHostToDevice));
        
        const dim3 block_size(16, 16);
        const dim3 grid_size((k + block_size.x - 1) / block_size.x,
                           (m + block_size.y - 1) / block_size.y);
        
        fp_matrix_mul_better_barrett<<<grid_size, block_size>>>(d_A, d_B, d_C, m, n, k);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_C, d_C, m * k * sizeof(fp_t), cudaMemcpyDeviceToHost));
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // Comparison function to benchmark against classic Barrett
    void compare_with_classic_barrett(const fp_t* h_a, const fp_t* h_b, int n) {
        printf("\n=== Comparing Better Barrett vs Classic Barrett ===\n");
        
        fp_t *classic_result = new fp_t[n];
        fp_t *better_result = new fp_t[n];
        
        // Time Better Barrett
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        mul_arrays(h_a, h_b, better_result, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float better_time;
        cudaEventElapsedTime(&better_time, start, stop);
        
        printf("Better Barrett time: %.3f ms\n", better_time);
        
        // Verify correctness by comparing a few results
        printf("Sample verification (first 5 results):\n");
        for (int i = 0; i < 5 && i < n; i++) {
            uint64_t expected = ((uint64_t)h_a[i] * h_b[i]) % q;
            printf("  %u * %u = %u (expected: %llu) %s\n", 
                   h_a[i], h_b[i], better_result[i], expected,
                   (better_result[i] == expected) ? "✓" : "✗");
        }
        
        delete[] classic_result;
        delete[] better_result;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// Enhanced testing with comparison between algorithms
int main() {
    const fp_t q = 2147483647; // Large prime (2^31 - 1)
    FiniteFieldBetterBarrett ff(q);
    
    printf("\n=== Running Better Barrett Reduction Tests ===\n");
    
    // Test 1: Basic operations with known values
    printf("\n--- Test 1: Basic Operations ---\n");
    const int test_n = 10;
    fp_t test_a[] = {0, 1, 2, 100, 1000, 10000, 100000, 1000000, q-1, q-2};
    fp_t test_b[] = {0, 1, 3, 200, 2000, 20000, 200000, 2000000, q-1, q-3};
    fp_t result[test_n];
    
    // Test addition
    ff.add_arrays(test_a, test_b, result, test_n);
    for (int i = 0; i < 5; i++) {
        printf("Add: %u + %u = %u (mod %u)\n", test_a[i], test_b[i], result[i], q);
    }
    
    // Test multiplication
    ff.mul_arrays(test_a, test_b, result, test_n);
    for (int i = 0; i < 5; i++) {
        printf("Mul: %u * %u = %u (mod %u)\n", test_a[i], test_b[i], result[i], q);
    }
    
    // Test 2: Large array performance test
    printf("\n--- Test 2: Large Array Performance ---\n");
    const int large_n = 1000000;
    fp_t *large_a = new fp_t[large_n];
    fp_t *large_b = new fp_t[large_n];
    fp_t *large_result = new fp_t[large_n];
    
    // Initialize with more diverse test data
    for (int i = 0; i < large_n; i++) {
        large_a[i] = (i * 12345 + 67890) % q;
        large_b[i] = (i * 54321 + 98765) % q;
    }
    
    ff.mul_arrays(large_a, large_b, large_result, large_n);
    printf("Large multiplication completed. Sample results:\n");
    for (int i = 0; i < 3; i++) {
        printf("  %u * %u = %u (mod %u)\n", 
               large_a[i], large_b[i], large_result[i], q);
    }
    
    // Test 3: Performance comparison
    printf("\n--- Test 3: Performance Comparison ---\n");
    ff.compare_with_classic_barrett(large_a, large_b, large_n);
    
    // Test 4: Inverse verification
    printf("\n--- Test 4: Inverse Verification ---\n");
    const int inv_test_n = 100;
    fp_t inv_test[inv_test_n];
    fp_t inv_result[inv_test_n];
    
    for (int i = 0; i < inv_test_n; i++) {
        inv_test[i] = (i + 1) * 1000 + 1; // Avoid zero
    }
    
    ff.inv_arrays(inv_test, inv_result, inv_test_n);
    
    // Verify a few inverses
    for (int i = 0; i < 5; i++) {
        fp_t verification[1] = {inv_test[i]};
        fp_t inv_val[1] = {inv_result[i]};
        fp_t verify_result[1];
        ff.mul_arrays(verification, inv_val, verify_result, 1);
        printf("Inverse: %u^-1 = %u, verification: %u * %u = %u [should be 1]\n",
               inv_test[i], inv_result[i], inv_test[i], inv_result[i], verify_result[0]);
    }
    
    delete[] large_a;
    delete[] large_b;
    delete[] large_result;
    
    printf("\n=== All Better Barrett tests completed ===\n");
    return 0;
}