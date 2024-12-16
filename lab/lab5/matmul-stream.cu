#include <cstdio>
#include <cuda.h>

int N = 1024;
const int nStreams = 4;
float *A, *B, *C;
float *dA, *dB, *dC;
cudaStream_t streams[nStreams];

// Kernel that performs the matrix vector multiplication b(i) = sum_j(A(i, j), x(j))
// A is row-major (stored row-by-row in memory)
__device__ void matvec(float *A, float *x, float *b, int n)
{
  // TODO / A FAIRE ...
}

int main()
{
  // A is stored by rows, A(i, j) = A[i * N + j]
  A = (float *) malloc (N * N * sizeof(float));
  // B and C are stored by columns, B(i, j) = B[i + j * N]
  B = (float *) malloc (N * N * sizeof(float));
  C = (float *) malloc (N * N * sizeof(float));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j; // A(i, j) = i + j
      B[i + j * N] = i - j; // B(i, j) = i - j
      C[i + j * N] = 0; // C(i, j) = 0
    }
  }
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dB, N * nStreams * sizeof(float));
  cudaMalloc(&dC, N * nStreams * sizeof(float));

  // Only copy the entire matrix A. For B and C, they need to be copied and computed one column vector at a time in a streaming manner
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Create streams
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Compute the matrix-vector multiplication C(:, j) = A * B(:, j) column-by-column using nStreams streams
  for (int j = 0; j < N; j++) {
    // Copy the column j of B into one of slots in dB using the stream no (j % nStreams) and cudaMemcpyAsync
    // TODO / A FAIRE ...

    // Perform the matrix-vector multiplication on A and the column vector in dB(:, j % nStreams), compute on dC(:, j % nStreams), using stream no (j % nStreams)
    // TODO / A FAIRE ...

    // Copy back the computed vector dC(:, j % nStreams) into the column C(:, j) using the same stream no (j % nStreams) and cudaMemcpyAsync
  }
  
  cudaDeviceSynchronize();

  free(A); free(B); free(C);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
