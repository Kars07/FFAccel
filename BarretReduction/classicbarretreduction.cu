#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <nvtx3/nvToolsExt.h>

// Data types for finite field elements
typedef uint32_t fp_t;     // Assumes p fits in 32 bits
typedef uint64_t fp_wide_t; // Wide type for intermediate calculations

// Device constants
__device__ __constant__ fp_t PRIME_P;
__device__ __constant__ fp_t MU;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Barrett reduction implementation 
__device__ __forceinline__ fp_t barrett_reduce(fp_wide_t x) {
    
    const int m = 32; // bit length of prime p (assuming 32-bit prime)
    fp_wide_t c = x >> (m - 1);
    
    fp_wide_t temp = c * MU;
    fp_wide_t quot = temp >> (m + 1);
    
    fp_wide_t rem = x - quot * PRIME_P;
    
    if (rem >= PRIME_P) {
        
        rem = rem - PRIME_P;

        if (rem >= PRIME_P) {

            rem = rem - PRIME_P;
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
    return barrett_reduce(prod);
}

__device__ __forceinline__ fp_t fp_neg_barrett(fp_t a) {
    return (a == 0) ? 0 : (PRIME_P - a);
}

// Modular exponentiation using Barrett reduction
__device__ fp_t fp_pow_barrett(fp_t base, fp_t exp) {
    fp_t result = 1;
    base = barrett_reduce(base); // Ensure base is within the field
    
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

// Host wrapper class with Barrett reduction
class FiniteFieldFpBarrett {
private:
    fp_t p;
    fp_t mu; // Precomputed Barrett constant
    
    // Compute Barrett constant μ = ⌊2^(2m)/p⌋
    fp_t compute_mu(fp_t prime) {
        // For 32-bit prime, 2m = 64
        // μ = ⌊2^64/p⌋
        // Using integer division to compute this
        uint64_t two_pow_64 = 0xFFFFFFFFFFFFFFFFULL; // 2^64 - 1 (approximation)
        return (fp_t)(two_pow_64 / prime);
    }
    
public:
    FiniteFieldFpBarrett(fp_t prime) : p(prime) {
        mu = compute_mu(prime);
        
        // Copy prime and mu to device constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(PRIME_P, &p, sizeof(fp_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(MU, &mu, sizeof(fp_t)));
        
        printf("Barrett reduction initialized with p = %u, μ = %u\n", p, mu);
    }
    
    void add_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
        nvtxRangePush("add_arrays_barrett");
        fp_t *d_a, *d_b, *d_result;

        nvtxRangePush("Memory Allocation");
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(fp_t)));
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        fp_add_arrays_barrett<<<grid_size, block_size>>>(d_a, d_b, d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
       
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePush("Memory Deallocation");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        nvtxRangePop();
        nvtxRangePop();
    }
    
    void mul_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
        nvtxRangePush("mul_arrays_barrett");
        fp_t *d_a, *d_b, *d_result;
        
        nvtxRangePush("Memory Allocation");
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(fp_t)));
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        fp_mul_arrays_barrett<<<grid_size, block_size>>>(d_a, d_b, d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePush("Memory Deallocation");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        nvtxRangePop();
        nvtxRangePop();
    }
    
    void inv_arrays(const fp_t* h_a, fp_t* h_result, int n) {
        nvtxRangePush("inv_arrays_barrett");
        fp_t *d_a, *d_result;
        
        nvtxRangePush("Memory Allocation");
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(fp_t)));
        CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(fp_t)));
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy H2D");
        CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(fp_t), cudaMemcpyHostToDevice));
        nvtxRangePop();
        
        nvtxRangePush("Kernel Execution");
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        fp_inv_arrays_barrett<<<grid_size, block_size>>>(d_a, d_result, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        nvtxRangePush("Memory Copy D2H");
        CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
        nvtxRangePop();
        
        nvtxRangePush("Memory Deallocation");
        cudaFree(d_a);
        cudaFree(d_result);
        nvtxRangePop();
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
        
        dim3 block_size(16, 16);
        dim3 grid_size((k + block_size.x - 1) / block_size.x,
                      (m + block_size.y - 1) / block_size.y);
        
        fp_matrix_mul_barrett<<<grid_size, block_size>>>(d_A, d_B, d_C, m, n, k);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_C, d_C, m * k * sizeof(fp_t), cudaMemcpyDeviceToHost));
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

// Example usage and testing
int main() {
    const fp_t p = 2147483647; // Large prime (2^31 - 1)
    FiniteFieldFpBarrett ff(p);
    
    const int n = 1000000;
    fp_t *a = new fp_t[n];
    fp_t *b = new fp_t[n];
    fp_t *result = new fp_t[n];
    
    // Initialize test data
    for (int i = 0; i < n; i++) {
        a[i] = i % p;
        b[i] = (i * 2) % p;
    }
    
    // Test addition
    ff.add_arrays(a, b, result, n);
    printf("Barrett Addition test: %u + %u = %u (mod %u)\n", a[0], b[0], result[0], p);
    
    // Test multiplication
    ff.mul_arrays(a, b, result, n);
    printf("Barrett Multiplication test: %u * %u = %u (mod %u)\n", a[0], b[0], result[0], p);
    
    // Test inverse (on smaller array for performance)
    const int small_n = 1000;
    ff.inv_arrays(a, result, small_n);
    printf("Barrett Inverse test: inv(%u) = %u (mod %u)\n", a[1], result[1], p);
    
    // Verify correctness by testing a[1] * result[1] ≡ 1 (mod p)
    fp_t verification[1] = {a[1]};
    fp_t inv_result[1] = {result[1]};
    fp_t verify_result[1];
    ff.mul_arrays(verification, inv_result, verify_result, 1);
    printf("Verification: %u * %u = %u (mod %u) [should be 1]\n", 
           a[1], result[1], verify_result[0], p);
    
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}