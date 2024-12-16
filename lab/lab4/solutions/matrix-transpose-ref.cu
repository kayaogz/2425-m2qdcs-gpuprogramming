/**
  * Kernel 1: Implement a first kernel where each block (using BSXY x BSXY threads) transposes a BSXY x BSXY tile of A, and writes it into the corresponding location in At. Do without using shared memory.
  *
  * Kernel 2: In the second kernel, do the same, but using the shared memory. Each block should load a tile of BSXY x BSXY of A into the shared memory, then perform the transposition using this tile in the shared memory into At. Test the difference in speedup. Test the performance using shared memory without padding and with padding (to avoid shared memory bank conflicts).
  *
  * Kernel 3: In this kernel, perform the transpose in-place on the matrix A (do not use At). A block should be transpose two tiles simultenously to be able to do this.
  *
  */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>
#include <chrono>


#define BSXY 32
float *dA, *dAt;
__global__ void transposeGPU(float *dA, float *dAt, int n)
{
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  if(i<n && j<n) dAt[i*n+j]=dA[j*n+i];
}
__global__ void transposeGPUSharedMemoryA(float *dA, float *dAt, int n)
{
  __shared__ float shAt[BSXY][BSXY];
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  if(i<n && j<n)
  {
    shAt[threadIdx.x][threadIdx.y]=dA[i*n+j];
    __syncthreads();
    i = threadIdx.y + blockIdx.y * blockDim.y;
    j = threadIdx.x + blockIdx.x * blockDim.x;
    dAt[i*n+j]=shAt[threadIdx.y][threadIdx.x];
  }
}
__global__ void transposeGPUSharedMemoryBankConflicts(float *dA, float *dAt, int n)
{
  __shared__ float shAt[BSXY][BSXY+1];
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  if(i<n && j<n)
  {
    shAt[threadIdx.x][threadIdx.y]=dA[i*n+j];
    __syncthreads();
    i = threadIdx.y + blockIdx.y * blockDim.y;
    j = threadIdx.x + blockIdx.x * blockDim.x;
    dAt[i*n+j]=shAt[threadIdx.y][threadIdx.x];
  }
}
__global__ void transposeGPUSharedMemoryInplaceA(float *dA, int n)
{
  if(blockIdx.x>=blockIdx.y)
  {
  __shared__ float shAt_a[BSXY][BSXY];
  __shared__ float shAt_b[BSXY][BSXY];
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  int i_1 = threadIdx.y + blockIdx.y * blockDim.y;
  int j_1 = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<n && j<n && i_1<n && j_1<n)
  {
    shAt_a[threadIdx.x][threadIdx.y]=dA[i*n+j];
    shAt_b[threadIdx.x][threadIdx.y]=dA[i_1*n+j_1];
    __syncthreads();
    dA[i*n+j]=shAt_b[threadIdx.y][threadIdx.x];
    dA[i_1*n+j_1]=shAt_a[threadIdx.y][threadIdx.x];
  }
  }
}
__global__ void transposeGPUSharedMemoryInplaceBankConflicts(float *dA, int n)
{
  if(blockIdx.x>=blockIdx.y)
  {
  __shared__ float shAt_a[BSXY][BSXY+1];
  __shared__ float shAt_b[BSXY][BSXY+1];
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  int i_1 = threadIdx.y + blockIdx.y * blockDim.y;
  int j_1 = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<n && j<n && i_1<n && j_1<n)
  {
    shAt_a[threadIdx.x][threadIdx.y]=dA[i*n+j];
    shAt_b[threadIdx.x][threadIdx.y]=dA[i_1*n+j_1];
    __syncthreads();
    dA[i*n+j]=shAt_b[threadIdx.y][threadIdx.x];
    dA[i_1*n+j_1]=shAt_a[threadIdx.y][threadIdx.x];
  }
  }
}

void transposeCPU(float *A, float *At, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      At[i * N + j] = A[j * N + i];
    }
  }
}
void verifyResults(float * At,float *Atemp,int n)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (std::abs(At[i*n+j] - Atemp[i*n+j]) > 1e-6) {
        std::cout << "Tronspose is incorrect for the element At[" << i << "][" << j << "]" << Atemp[i * n + j]<<"|"<< At[i*n+j]<< std::endl;
        return;
      }
    }
  }
  std::cout << "Tronspose is correct!" << std::endl;
}
int main()
{
  // Allocate A and At
  // A is an N * N matrix stored by rows, i.e. A(i, j) = A[i * N + j]
  // At is also stored by rows and is the transpose of A, i.e., At(i, j) = A(j, i)
  int N = 1024;
  float *A = (float *) malloc(N * N * sizeof(A[0]));
  float *At = (float *) malloc(N * N * sizeof(At[0]));
  float *Atemp = (float *) malloc(N * N * sizeof(At[0]));
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      A[i*N+j]=i*N+j;

  transposeCPU(A,At,N);

  // Allocate dA and dAt, and call the corresponding matrix transpose kernel
  // TODO ...
  cudaMalloc(&dA, sizeof(dA[0]) * N * N);
  cudaMalloc(&dAt, sizeof(dAt[0]) * N * N);
  
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x=(N-1)/BSXY + 1;
    dimGrid.y=(N-1)/BSXY + 1;
    dimGrid.z = 1;
    dimBlock.x=BSXY;
    dimBlock.y=BSXY;
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto start=std::chrono::high_resolution_clock::now();
    transposeGPU<<<dimGrid, dimBlock>>>(dA,dAt,N);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto end=std::chrono::high_resolution_clock::now();
    cudaMemcpy(Atemp, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::chrono::duration<double> duration = end-start;
    printf("transposeGPU %f s\n",duration.count());
  }
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x=(N-1)/BSXY + 1;
    dimGrid.y=(N-1)/BSXY + 1;
    dimGrid.z = 1;
    dimBlock.x=BSXY;
    dimBlock.y=BSXY;
    dimBlock.z=1;
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++)
      transposeGPUSharedMemoryA<<<dimGrid, dimBlock>>>(dA,dAt,N);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto end=std::chrono::high_resolution_clock::now();
    cudaMemcpy(Atemp, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(At,Atemp,N);
    std::chrono::duration<double> duration = end-start;
    printf("transposeGPUSharedMemoryA %f s\n",duration.count()/N);
  }
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x=(N-1)/BSXY + 1;
    dimGrid.y=(N-1)/BSXY + 1;
    dimGrid.z = 1;
    dimBlock.x=BSXY;
    dimBlock.y=BSXY;
    dimBlock.z=1;
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++)
      transposeGPUSharedMemoryBankConflicts<<<dimGrid, dimBlock>>>(dA,dAt,N);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto end=std::chrono::high_resolution_clock::now();
    cudaMemcpy(Atemp, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(At,Atemp,N);
    std::chrono::duration<double> duration = end-start;
    printf("transposeGPUSharedMemoryBankConflicts %f s\n",duration.count()/N);
  }
  { 
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = (N-1)/BSXY + 1;
    dimGrid.y = (N-1)/BSXY + 1;
    dimGrid.z = 1;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N-1;i++)
      transposeGPUSharedMemoryInplaceA<<<dimGrid, dimBlock>>>(dA,N);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto end=std::chrono::high_resolution_clock::now();
    cudaMemcpy(Atemp, dA, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(At,Atemp,N);
    std::chrono::duration<double> duration = end-start;
    printf("transposeGPUSharedMemoryInplaceA %f s\n",duration.count()/(N-1));
  }
  { 
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = (N-1)/BSXY + 1;
    dimGrid.y = (N-1)/BSXY + 1;
    dimGrid.z = 1;
    dimBlock.x = BSXY; 
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N-1;i++)
      transposeGPUSharedMemoryInplaceBankConflicts<<<dimGrid, dimBlock>>>(dA,N);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto end=std::chrono::high_resolution_clock::now();
    cudaMemcpy(Atemp, dA, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(At,Atemp,N);
    std::chrono::duration<double> duration = end-start;
    printf("transposeGPUSharedMemoryInplaceBankConflicts %f s\n",duration.count()/(N-1));
  }
  // Deallocate dA and dAt
  // TODO ...
  cudaFree(dA);
  cudaFree(dAt);
  // Deallocate A and At
  free(A);
  free(At);
  return 0;
}
