#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

__global__ void cudaCopieParBlocs(float *tab0, const float *tab1, int taille)
{
  int idx = blockIdx.x;
  if (idx < taille) { tab0[idx] = tab1[idx]; }
}

__global__ void cudaCopieParBlocsThreads(float *tab0, const float *tab1, int taille)
{
  // Calculer le bon idx en fonction du blockIdx.x, threadIdx.x, et blockDim.x
  // A FAIRE
  int idx; // idx = ?
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < taille) { tab0[idx] = tab1[idx]; }
}

int main(int argc, char **argv) {
  float *A, *B, *Ad, *Bd;
  int N, i;

  if (argc < 2) {
    printf("Utilisation: ./cuda-copie N\n");
    return 0;
  }
  N = atoi(argv[1]);

  // Initialisation
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { 
    A[i] = (float)i;
    B[i] = 0.0f;
  }
  
  // Allouer les tableau Ad et Bd dynamiques de taille N sur le GPU avec cudaMalloc 
  // A FAIRE
  cudaMalloc(&Ad, N * sizeof(float));
  cudaMalloc(&Bd, N * sizeof(float));

  // Copier A dans Ad et B dans Bd
  // A FAIRE
  cudaMemcpy(Ad, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bd, B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Copier Ad dans Bd avec le kernel cudaCopieParBlocs
  // A FAIRE ...
  cudaCopieParBlocs<<<N, 1>>>(Bd, Ad, N);

  // Attendre que le kernel cudaCopieParBlocs termine
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verifier le resultat en CPU en copiant Bd dans B puis en comparant B avec A
  // A FAIRE
  cudaMemcpy(B, Bd, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Remettre B et Bd a zero pour tester le deuxieme kernel de copie
  // A FAIRE
  for (int i = 0; i < N; i++) { B[i] = 0.0f; }
  cudaMemcpy(Bd, B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Copier Ad dans Bd avec le kernel cudaCopieParBlocsThreads
  // A FAIRE ...
  cudaCopieParBlocsThreads<<<(N-1)/1024 + 1, 1024>>>(Bd, Ad, N);

  // Attendre que le kernel cudaCopieParBlocsThreads termine
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verifier le resultat en CPU en copiant Bd dans B puis en comparant B avec A
  // A FAIRE
  cudaMemcpy(B, Bd, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Desaollouer le tableau Ad[N] et Bd[N] sur le GPU
  // A FAIRE ...
  cudaFree(Ad);
  cudaFree(Bd);

  // Desallouer A et B
  free(A);
  free(B);

  return 0;
  }
