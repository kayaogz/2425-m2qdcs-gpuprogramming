#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

int main() {
  float *A, *B, *Ad;
  int N, i;

  if (argc < 2) {
    printf("Utilisation: ./cuda-copie-dynamique N\n");
    return 0;
  }
  N = atoi(argv[1]);

  // Initialisation
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { A[i] = (float)i; }
  
  // Allouer le tableau Ad dynamique de taille N sur le GPU avec cudaMalloc 
  // A FAIRE ...
  cudaMalloc(&Ad, N * sizeof(float));

  // cudaMemcpy de A[N] vers Ad[N]
  // A FAIRE ...
  cudaMemcpy(Ad, A, sizeof(float) * N, cudaMemcpyHostToDevice);

  // cudaMemcpy de Ad[N] vers B[N]
  // A FAIRE ...
  cudaMemcpy(B, Ad, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Desaollouer le tableau Ad[N] sur le GPU
  // A FAIRE ...
  cudaFree(Ad);

  // Attendre que les kernels GPUs terminent
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verifier le resultat
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }
  free(A);
  free(B);

  return 0;
}
