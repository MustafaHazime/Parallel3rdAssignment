#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__ void matrixMultiply(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

double seconds() {
    struct timeval tmp;
    gettimeofday(&tmp, NULL);
    return tmp.tv_sec + tmp.tv_usec / 1000000.0;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <M> <K> <N>\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Allocate memory on the host
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy input matrices from host to GPU
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block dimensions
    dim3 blockDim(16, 16);
    
    // Calculate grid dimensions
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);


    // Start the timer
    double start = seconds();

    // Launch the kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // Synchronize to make sure the kernel is finished
    cudaDeviceSynchronize();

    // Stop the timer
    double end = seconds();

    // Copy the result matrix from GPU to host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

        // Calculate the runtime and speedup
    double cudaRuntime = end - start;
    double sequentialRuntime = 0.0; // Variable to store sequential runtime

    // Start the timer for sequential execution
    start = seconds();

    // Perform sequential matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    // Stop the timer for sequential execution
    end = seconds();

    sequentialRuntime = end - start;

      // Print the result matrix
    printf("Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }


    // Print the runtime and speedup
    printf("CUDA Runtime: %.6f seconds\n", cudaRuntime);
    printf("Sequential Runtime: %.6f seconds\n", sequentialRuntime);
    printf("Speedup: %.2f\n", sequentialRuntime / cudaRuntime);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}

