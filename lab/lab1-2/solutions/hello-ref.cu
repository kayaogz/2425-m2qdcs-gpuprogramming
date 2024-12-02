#include <cstdio>
#include "cuda.h"

__global__ void cudaHello(){
  // Afficher le message Hello World ainsi que blockidx et threadidx depuis chaque thread
  // A FAIRE ...
  printf("Hello World from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

int main() {
  int numBlocks = 64;
  int blockSize = 1;
  cudaHello<<<numBlocks, blockSize>>>(); 

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  return 0;
  }
