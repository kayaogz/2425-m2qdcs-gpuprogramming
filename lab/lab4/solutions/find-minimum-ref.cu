#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>
#include <chrono>
#define BLOCKSIZE 1024

/**
  * Version 1: Ecrire un kernel GPU 1D qui trouve l'element minimum d'un tableau dA[N] pour chaque bloc et ecrit le minimum de chaque bloc dans une case de dAmin. En suite, CPU reprend dAmin et calcul le minimum global en sequentiel sur ce petit tableau.
  *
  * Version 2: Le premier appel au findMinimum reduit la taille du tableau a parcourir en sequentiel a N/BLOCKSIZE. Dans cette version, utiliser findMinimum deux fois a la suite afin de reduire la taille du tableau a parcourir en sequentiel a N/(BLOCKSIZE*BLOCKSIZE) (pour que la partie sequentielle en CPU devient vraiment negligable).
  *
  * Pour trouver le minimum des deux flottants en GPU, utiliser la fonction fminf(x, y)
  */

__global__ void findMinimum(float *dA, float *dAmin, int N)
{
  __shared__ volatile float buff[BLOCKSIZE];
  int idx = threadIdx.x + blockIdx.x * BLOCKSIZE;
  if (idx < N)
    buff[threadIdx.x] = dA[idx]; 
  else
    buff[threadIdx.x] = 0.0;
  if (BLOCKSIZE > 512) {
  __syncthreads(); 
  if (threadIdx.x < 512) 
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 512], buff[threadIdx.x]);
  else
    return; 
  }
  if (BLOCKSIZE > 256) {
  __syncthreads(); 
  if (threadIdx.x < 256) 
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 256], buff[threadIdx.x]);
  else
    return; 
  }
  if (BLOCKSIZE > 128) {
  __syncthreads(); 
  if (threadIdx.x < 128) 
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 128], buff[threadIdx.x]);
  else
    return; 
  }
  if (BLOCKSIZE > 64) {
  __syncthreads(); 
  if (threadIdx.x < 64) 
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 64], buff[threadIdx.x]);
  else
    return; 
  }
  if (BLOCKSIZE > 32) {
  __syncthreads(); 
  if (threadIdx.x < 32) 
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 32], buff[threadIdx.x]);
  else
    return; 
  }

  if (threadIdx.x < 16) {
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 16], buff[threadIdx.x]);}
  if (threadIdx.x < 8) {
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 8], buff[threadIdx.x]);}
  if (threadIdx.x < 4) {
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 4], buff[threadIdx.x]);}
  if (threadIdx.x < 2) {
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 2], buff[threadIdx.x]);}
  if (threadIdx.x < 1) {
    buff[threadIdx.x] = fminf(buff[threadIdx.x + 1], buff[threadIdx.x]);
    dAmin[blockIdx.x] = buff[threadIdx.x];
    }
}

using namespace std;

int main()
{
  srand(1234);
  int N = 100000000;
  int numBlocks=(N-1)/BLOCKSIZE + 1;
  float *A, *dA; // Le tableau dont minimum on va chercher
  float *Amin,*Amin1, *dAmin,*dAmin1; // Amin contiendra en suite le tableau reduit par un facteur de BLOCKSIZE apres l'execution du kernel GPU
  
  // Allour les tableaux A[N] et Amin[numBlocks] de maniere ``pined'' sur le CPU
  // Allouer les tableaux dA[N] et dAmin[numBlocks] sur le GPU
  // A FAIRE ...

  A = (float *) malloc(N * sizeof(A[0]));
  Amin=(float *) malloc(numBlocks * sizeof(Amin[0]));
  
  cudaMalloc(&dA, sizeof(dA[0]) * N );
  cudaMalloc(&dAmin, sizeof(dAmin[0]) * numBlocks );
  int numBlocks_1=(numBlocks-1)/BLOCKSIZE + 1;
  Amin1=(float *) malloc(numBlocks_1 * sizeof(Amin1[0]));
  cudaMalloc(&dAmin1, sizeof(dAmin[0]) * numBlocks_1 );

  // Initialiser le tableau A
  for (int i = 0; i < N; i++) { A[i] = (float)(rand() % 1000); }
  A[rand() % N] = -1.0; // Mettre le minimum a -1.

  // Mettre A sur le GPU (dA) avec memcpy
  // A FAIRE ...
  
  cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
  float minA = FLT_MAX; // Affecter le maximum float a minA
  // Trouver le minimum du tableau dA, mettre dAmin dans le CPU, puis trouver le minimum global et le mettre dans la variable minA
  // A FAIRE ...
  auto start=std::chrono::high_resolution_clock::now();
  findMinimum<<<numBlocks,BLOCKSIZE>>>(dA,dAmin,N);
  cudaMemcpy(Amin, dAmin, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
   for(int i=0;i<numBlocks;i++)
    if(minA>Amin[i])minA=Amin[i];
  auto end=std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end-start;
  if (minA == -1) { cout << "Version 1 The minimum is correct! " << duration.count() <<" s"<<endl; }
  else { cout << "Version 1 The minimum found (" << minA << ") is incorrect (it should have been -1)!" << endl; }
  cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
  auto start1=std::chrono::high_resolution_clock::now();
  findMinimum<<<numBlocks,BLOCKSIZE>>>(dA,dAmin,N);
  findMinimum<<<numBlocks_1,BLOCKSIZE>>>(dAmin,dAmin1,numBlocks);
  cudaMemcpy(Amin1, dAmin1, numBlocks_1 * sizeof(float), cudaMemcpyDeviceToHost);
  float minA1 = FLT_MAX;
  for(int i=0;i<numBlocks_1;i++)
    if(minA1>Amin1[i])minA1=Amin1[i];
  // Verifier le resultat
  auto end1=std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration1 = end1-start1;
  if (minA1 == -1) { cout << "Version 2 The minimum is correct! " << duration1.count() <<" s"<< endl; }
  else { cout << "Version 2 The minimum found (" << minA1 << ") is incorrect (it should have been -1)!" << endl; }

  return 0;
}
