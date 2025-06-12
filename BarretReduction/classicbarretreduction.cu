#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <nvtx3/nvToolsExt.h>

// Data types for finite field elements
typedef uint32_t fp_t;     // Assumes p fits in 32 bits
typedef uint64_t fp_wide_t; // Wide type for intermediate calculations

// Device constants
__device__ __constant__ fp_t PRIME_P;
__device__ __constant__ uint64_t MU;
__device__ __constant__ int M;  // Bit length of prime

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

//Barrett reduction implementation 
__device__ __forceinline__ fp_t barrett_reduce(fp_wide_t x) {
    // Step 1: c ← ⌊x / 2^(m-1)⌋
    fp_wide_t c = x >> (M - 1);
    
    // Step 2: quot ← ⌊(c * μ) / 2^(m+1)⌋
    // We need to be careful with overflow here
    // Use 128-bit arithmetic or split the multiplication
    fp_wide_t temp = c * MU;
    fp_wide_t quot = temp >> (M + 1);
    
    // Step 3: rem ← x - quot * q
    fp_wide_t rem = x - quot * PRIME_P;
    
    // Step 4-7: Final corrections
    if (rem >= PRIME_P) {
        rem = rem - PRIME_P;
        if (rem >= PRIME_P) {
            rem = rem - PRIME_P;
        }
    }
    
    return (fp_t)rem;
}

// Optimized Barrett reduction with precomputed shifts
__device__ __forceinline__ fp_t barrett_reduce_optimized(fp_wide_t x) {
    // For p = 2^31 - 1, m = 31
    // c = x >> 30 (m-1 = 30)
    fp_wide_t c = x >> 30;
    
    // quot = (c * μ) >> 32 (m+1 = 32)
    fp_wide_t temp = c * MU;
    fp_wide_t quot = temp >> 32;
    
    // rem = x - quot * p
    fp_wide_t rem = x - quot * PRIME_P;
    
    // Final reductions 
    if (rem >= PRIME_P) {
        rem -= PRIME_P;
        if (rem >= PRIME_P) {
            rem -= PRIME_P;
        }
    }
    
    return (fp_t)rem;
}

// Basic finite field operations using Barrett reduction
__device__ __forceinline__ fp_t fp_add_barrett(fp_t a, fp_t b) {
    fp_wide_t sum = (fp_wide_t)a + b;
    return (sum >= PRIME_P) ? (sum - PRIME_P) : (fp_t)sum;
}

__device__ __forceinline__ fp_t fp_sub_barrett(fp_t a, fp_t b) {
    return (a >= b) ? (a - b) : (a + PRIME_P - b);
}

__device__ __forceinline__ fp_t fp_mul_barrett(fp_t a, fp_t b) {
    fp_wide_t prod = (fp_wide_t)a * b;
    return barrett_reduce_optimized(prod);
}

__device__ __forceinline__ fp_t fp_neg_barrett(fp_t a) {
    return (a == 0) ? 0 : (PRIME_P - a);
}

// Modular exponentiation using Barrett reduction
__device__ fp_t fp_pow_barrett(fp_t base, fp_t exp) {
    fp_t result = 1;
    base = barrett_reduce_optimized(base); // Ensure base is within the field
    
    while (exp > 0) {
        if (exp & 1) {
            result = fp_mul_barrett(result, base);
        }
        exp >>= 1;
        base = fp_mul_barrett(base, base);
    }
    return result;
}

// Extended Euclidean Algorithm for modular inverse
__device__ fp_t fp_inv_barrett(fp_t a) {
    if (a == 0) {
        return 0; // Invalid input
    }

    int64_t old_r = PRIME_P, r = a;
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
    
    return (old_s < 0) ? (fp_t)(old_s + PRIME_P) : (fp_t)old_s;
}

// Fermat's Little Theorem for modular inverse using Barrett reduction
__device__ fp_t fp_inv_fermat_barrett(fp_t a) {
    if (a == 0) {
        return 0; // Invalid input
    }
    return fp_pow_barrett(a, PRIME_P - 2);
}

__device__ __forceinline__ fp_t fp_div_barrett(fp_t a, fp_t b) {
    return fp_mul_barrett(a, fp_inv_barrett(b));
}

// Batch operations kernels using Barrett reduction
__global__ void fp_add_arrays_barrett(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_add_barrett(a[idx], b[idx]);
    }
}

__global__ void fp_sub_arrays_barrett(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_sub_barrett(a[idx], b[idx]);
    }
}

__global__ void fp_mul_arrays_barrett(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_mul_barrett(a[idx], b[idx]);
    }
}

__global__ void fp_pow_arrays_barrett(const fp_t* base, const fp_t* exp, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_pow_barrett(base[idx], exp[idx]);
    }
}

__global__ void fp_inv_arrays_barrett(const fp_t* a, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_inv_barrett(a[idx]);
    }
}

// Polynomial evaluation using Barrett reduction
__global__ void fp_poly_eval_barrett(
    const fp_t* coeffs, int degree,
    const fp_t* x_vals, fp_t* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp_t x = x_vals[idx];
        fp_t result = coeffs[degree];

        for (int i = degree - 1; i >= 0; i--) {
            result = fp_add_barrett(fp_mul_barrett(result, x), coeffs[i]);
        }
        results[idx] = result;
    }
}

// Matrix multiplication using Barrett reduction
__global__ void fp_matrix_mul_barrett(const fp_t* A, const fp_t* B, fp_t* C,
                                     int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        fp_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum = fp_add_barrett(sum, fp_mul_barrett(A[row * n + i], B[i * k + col]));
        }
        C[row * k + col] = sum;
    }
}

// Host wrapper class with corrected Barrett reduction
class FiniteFieldFpBarrett {
private:
    fp_t p;
    uint64_t mu;
    int m; // Bit length of prime
    
    //compute Barrett constant μ = ⌊2^(2m)/p⌋
    uint64_t compute_mu(fp_t prime) {
        // Find bit length of prime (number of bits needed to represent prime)
        int bit_length = 0;
        fp_t temp = prime;
        while (temp > 0) {
            bit_length++;
            temp >>= 1;
        }
        
        // For Barrett reduction: μ = ⌊2^(2m)/p⌋
        // We need to compute this carefully to avoid overflow
        
        // Method 1: Use long double for higher precision
        long double numerator = 1.0L;
        for (int i = 0; i < 2 * bit_length; i++) {
            numerator *= 2.0L;
        }
        
        uint64_t mu_result = (uint64_t)(numerator / (long double)prime);
        
        // Method 2: Alternative calculation using bit shifts (for verification)
        // For p = 2^31 - 1, we need 2^62 / p
        // Since 2^62 is too large for uint64_t, we use the approximation
        
        printf("Prime bit length: %d\n", bit_length);
        printf("Computing μ = ⌊2^%d / %u⌋\n", 2 * bit_length, prime);
        
        return mu_result;
    }
    
public:
    FiniteFieldFpBarrett(fp_t prime) : p(prime) {
        // Compute bit length
        m = 0;
        fp_t temp = prime;
        while (temp > 0) {
            m++;
            temp >>= 1;
        }
        
        mu = compute_mu(prime);
        
        // Copy constants to device memory
        CUDA_CHECK(cudaMemcpyToSymbol(PRIME_P, &p, sizeof(fp_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(MU, &mu, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(M, &m, sizeof(int)));
        
        printf("Barrett reduction initialized:\n");
        printf("  p = %u\n", p);
        printf("  m = %d (bit length)\n", m);
        printf("  μ = %llu\n", (unsigned long long)mu);
        
        // Verify μ calculation
        long double expected = 1.0L;
        for (int i = 0; i < 2 * m; i++) {
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
        nvtxRangePush("add_arrays_barrett");
        
        DeviceArrays arrays(n);
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(arrays.d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(arrays.d_b, h_b, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        fp_add_arrays_barrett<<<grid_size, block_size>>>(arrays.d_a, arrays.d_b, arrays.d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
       
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, arrays.d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePop();
    }
    
    void mul_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
        nvtxRangePush("mul_arrays_barrett");
        
        DeviceArrays arrays(n);
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(arrays.d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(arrays.d_b, h_b, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        fp_mul_arrays_barrett<<<grid_size, block_size>>>(arrays.d_a, arrays.d_b, arrays.d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, arrays.d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePop();
    }
    
    void inv_arrays(const fp_t* h_a, fp_t* h_result, int n) {
        nvtxRangePush("inv_arrays_barrett");
        
        fp_t *d_a, *d_result;
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(fp_t)));
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        fp_inv_arrays_barrett<<<grid_size, block_size>>>(d_a, d_result, n);
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
        
        fp_matrix_mul_barrett<<<grid_size, block_size>>>(d_A, d_B, d_C, m, n, k);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_C, d_C, m * k * sizeof(fp_t), cudaMemcpyDeviceToHost));
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

// Enhanced testing with more comprehensive verification
int main() {
    const fp_t p = 2147483647; // Large prime (2^31 - 1)
    FiniteFieldFpBarrett ff(p);
    
    printf("\n=== Running Comprehensive Barrett Reduction Tests ===\n");
    
    // Test 1: Basic operations with known values
    printf("\n--- Test 1: Basic Operations ---\n");
    const int test_n = 10;
    fp_t test_a[] = {0, 1, 2, 100, 1000, 10000, 100000, 1000000, p-1, p-2};
    fp_t test_b[] = {0, 1, 3, 200, 2000, 20000, 200000, 2000000, p-1, p-3};
    fp_t result[test_n];
    
    // Test addition
    ff.add_arrays(test_a, test_b, result, test_n);
    for (int i = 0; i < 5; i++) {
        printf("Add: %u + %u = %u (mod %u)\n", test_a[i], test_b[i], result[i], p);
    }
    
    // Test multiplication
    ff.mul_arrays(test_a, test_b, result, test_n);
    for (int i = 0; i < 5; i++) {
        printf("Mul: %u * %u = %u (mod %u)\n", test_a[i], test_b[i], result[i], p);
    }
    
    // Test 2: Large array performance test
    printf("\n--- Test 2: Large Array Performance ---\n");
    const int large_n = 1000000;
    fp_t *large_a = new fp_t[large_n];
    fp_t *large_b = new fp_t[large_n];
    fp_t *large_result = new fp_t[large_n];
    
    // Initialize with more diverse test data
    for (int i = 0; i < large_n; i++) {
        large_a[i] = (i * 12345 + 67890) % p;
        large_b[i] = (i * 54321 + 98765) % p;
    }
    
    ff.mul_arrays(large_a, large_b, large_result, large_n);
    printf("Large multiplication completed. Sample results:\n");
    for (int i = 0; i < 3; i++) {
        printf("  %u * %u = %u (mod %u)\n", 
               large_a[i], large_b[i], large_result[i], p);
    }
    
    // Test 3: Inverse verification
    printf("\n--- Test 3: Inverse Verification ---\n");
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
    
    printf("\n=== All tests completed ===\n");
    return 0;
}