#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <nvtx3/nvToolsExt.h>


//Data types for finite field elements
typedef uint32_t fp_t; // Assumes p fits in 32 bits
typedef uint32_t fp_wide_t; // Wide type for intermediate calculations

//Device Constant for the prime p
__device__ __constant__ fp_t  PRIME_P;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

//Basic finite field operatioins
__device__ __forceinline__ fp_t fp_add(fp_t a, fp_t b) {
    fp_wide_t sum = (fp_wide_t)a + b;
    return (sum >= PRIME_P) ? (sum - PRIME_P) : (fp_t)sum;
}

__device__ __forceinline__ fp_t fp_sub(fp_t a, fp_t b) {
    return (a >= b) ? (a - b) : (a + PRIME_P - b);
}

__device__ __forceinline__ fp_t fp_mul(fp_t a, fp_t b) {
    fp_wide_t prod = (fp_wide_t)a * b;
    return (fp_t)(prod % PRIME_P);
}

__device__ __forceinline__ fp_t fp_neg(fp_t a) {
    return (a == 0) ? 0 : (PRIME_P - a);
}

//Modular Exponentiation using binary method
__device__ fp_t fp_pow(fp_t base, fp_t exp) {
    fp_t result = 1;
    base = base % PRIME_P; // Ensure base is within the field
    
    while (exp > 0)
    {
        if (exp & 1) {
            result = fp_mul(result, base);
        }
        exp >>= 1;
        base = fp_mul(base, base);
    }
    return result;
}

//Extended Euclidean Algorithm to find modular inverse
__device__ fp_t fp_inv(fp_t a) {
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
    // old_r should be the GCD, which is 1 if a is invertible, and old_s is the inverse
    return (old_s < 0) ? (fp_t)(old_s + PRIME_P) : (fp_t)old_s;
}

//Fermat's Little Theorem for modular inverse
__device__ fp_t fp_inv_fermat(fp_t a) {
    if (a == 0) {
        return 0; // Invalid input
    }
    return fp_pow(a, PRIME_P - 2);
}

__device__ __forceinline__ fp_t fp_div(fp_t a, fp_t b) {
    return fp_mul(a, fp_inv(b));
}

//Batch operations kernels
__global__ void fp_add_arrays(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_add(a[idx], b[idx]);
    }
}
__global__ void fp_sub_arrays(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_sub(a[idx], b[idx]);
    }
}
__global__ void fp_mul_arrays(const fp_t* a, const fp_t* b, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_mul(a[idx], b[idx]);
    }
}
__global__ void fp_pow_arrays(const fp_t* base, const fp_t* exp, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_pow(base[idx], exp[idx]);
    }
}
__global__ void fp_inv_arrays(const fp_t* a, fp_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fp_inv(a[idx]);
    }
}

//Polynomial evaluation in Fp using Horners method
__global__ void fp_poly_eval(
const fp_t* coeffs, int degree,
const fp_t* x_vals, fp_t* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp_t x = x_vals[idx];
        fp_t result = coeffs[degree];;

        for (int i = degree - 1; i >= 0; i--) {
            result = fp_add(fp_mul(result, x), coeffs[i]);
        }
            results[idx] = result; 
    }
}

//Matrix operations in Fp
__global__ void fp_matrix_add(const fp_t* A, const fp_t* B, fp_t* C, int m, int n, int k) {
    int row = blockIdx.y + blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        fp_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum = fp_add(sum, fp_mul(A[row * n + i], B[i * k + col]));
        }
        C[row * k + col] = sum;
    }
}
__global__ void fp_matrix_mul(const fp_t* A, const fp_t* B, fp_t* C,
                              int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        fp_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum = fp_add(sum, fp_mul(A[row * n + i], B[i * k + col]));
        }
        C[row * k + col] = sum;
    }
}

//Host wrapper Functions
class FiniteFieldFp {
    private:
        fp_t p;
    public:
        FiniteFieldFp(fp_t prime) : p(prime) {
            //Copy prime to device constant memory
            CUDA_CHECK(cudaMemcpyToSymbol(PRIME_P, &p, sizeof(fp_t)));
        }
        void add_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
            nvtxRangePush("add_arrays");
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
            fp_add_arrays<<<grid_size, block_size>>>(d_a, d_b, d_result, n);
            nvtxRangePop();
           
            nvtxRangePush("Memory Copy D2H");
            CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
            nvtxRangePop();
            
            nvtxRangePush("Memory Deallocation");
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_result);
            nvtxRangePop();
            nvtxRangePop(); // add_arrays
        }
        void mul_arrays(const fp_t* h_a, const fp_t* h_b, fp_t* h_result, int n) {
            nvtxRangePush("mul_arrays");
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
            fp_mul_arrays<<<grid_size, block_size>>>(d_a, d_b, d_result, n);
            nvtxRangePop();
            
            nvtxRangePush("Memory Copy D2H");
            CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
            nvtxRangePop();
            
            nvtxRangePush("Memory Deallocation");
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_result);
            nvtxRangePop();
            nvtxRangePop(); // mul_arrays
        }
        
        void inv_arrays(const fp_t* h_a, fp_t* h_result, int n) {
            nvtxRangePush("inv_arrays");
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
            fp_inv_arrays<<<grid_size, block_size>>>(d_a, d_result, n);
            nvtxRangePop();
            
            nvtxRangePush("Memory Copy D2H");
            CUDA_CHECK(cudaMemcpy(h_result, d_result, n * sizeof(fp_t), cudaMemcpyDeviceToHost));
            nvtxRangePop();
            
            nvtxRangePush("Memory Deallocation");
            cudaFree(d_a);
            cudaFree(d_result);
            nvtxRangePop();
            nvtxRangePop(); // inv_arrays
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
            
            fp_matrix_mul<<<grid_size, block_size>>>(d_A, d_B, d_C, m, n, k);
            
            CUDA_CHECK(cudaMemcpy(h_C, d_C, m * k * sizeof(fp_t), cudaMemcpyDeviceToHost));
            
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
};
// Example usage and testing
int main() {
    const fp_t p = 2147483647; // Large prime (2^31 - 1)
    FiniteFieldFp ff(p);
    
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
    printf("Addition test: %u + %u = %u (mod %u)\n", a[0], b[0], result[0], p);
    
    // Test multiplication
    ff.mul_arrays(a, b, result, n);
    printf("Multiplication test: %u * %u = %u (mod %u)\n", a[0], b[0], result[0], p);
    
    // Test inverse (on smaller array for performance)
    const int small_n = 1000;
    ff.inv_arrays(a, result, small_n);
    printf("Inverse test: inv(%u) = %u (mod %u)\n", a[1], result[1], p);
    
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}