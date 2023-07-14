#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
// #include <helper_functions.h>

#define VECTORDIM 3
typedef float mt;

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void DistanceCPU(mt *array1, mt *array2, int narray1, int narray2, mt *output) {
  mt temp;
  for (int i = 0; i < narray1; i++) {
    for (int j = 0; j < narray2; j++) {
      temp = 0;
      for (int l = 0; l < VECTORDIM; l++) {
#ifndef USE_POW
        temp += (array1[i + l * narray1] - array2[j + l * narray2]) *
                (array1[i + l * narray1] - array2[j + l * narray2]);
#else
        temp += powf(array1[i + l * narray1] - array2[j + l * narray2], 2);
#endif
      }
      output[i * narray2 + j] = temp;
    }
  }
}
__global__ void DistGPU(mt *array1, mt *array2, int narray1, int narray2,
                        mt *output) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  mt temp;

  if (i < narray1) {
    for (int j = 0; j < narray2; j++) {
      temp = 0;
#ifndef USE_POW
      temp += (array1[i] - array2[j]) * (array1[i] - array2[j]);
      temp += (array1[i + narray1] - array2[j + narray2]) *
              (array1[i + narray1] - array2[j + narray2]);
      temp += (array1[i + 2 * narray1] - array2[j + 2 * narray2]) *
              (array1[i + 2 * narray1] - array2[j + 2 * narray2]);
#else
      temp += powf(array1[i] - array2[j], 2);
      temp += powf(array1[i + narray1] - array2[j + narray2], 2);
      temp += powf(array1[i + 2 * narray1] - array2[j + 2 * narray2], 2);
#endif
      output[i * narray2 + j] = temp;
    }
  }
}

int main() {
  int narray1 = 7000;
  int narray2 = 60000;

  mt *array1 = new mt[narray1 * VECTORDIM];
  mt *array2 = new mt[narray2 * VECTORDIM];
  mt *outputGPU = new mt[narray1 * narray2];
  mt *outputCPU = new mt[narray1 * narray2];
  mt *outputCPUTest = new mt[narray1 * narray2];

  mt *d_array1;
  mt *d_array2;
  mt *d_output;

  for (int i = 0; i < narray1 * VECTORDIM; i++) {
    array1[i] = static_cast<mt>(rand() / (static_cast<mt>(RAND_MAX / 10)));
    // std::cout << "Element " << i << " " << array1[i] << std::endl;
  }

  for (int i = 0; i < narray2 * VECTORDIM; i++) {
    array2[i] = static_cast<mt>(rand() / (static_cast<mt>(RAND_MAX / 10)));
  }

  cudaError_t err;

  err = cudaMalloc((void **)&d_array1, narray1 * VECTORDIM * sizeof(mt));
  err = cudaMalloc((void **)&d_array2, narray2 * VECTORDIM * sizeof(mt));
  err = cudaMalloc((void **)&d_output, narray1 * narray2 * sizeof(mt));

  err = cudaMemcpy(d_array1, array1, narray1 * VECTORDIM * sizeof(mt),
                   cudaMemcpyHostToDevice);
  err = cudaMemcpy(d_array2, array2, narray2 * VECTORDIM * sizeof(mt),
                   cudaMemcpyHostToDevice);

  int threadsPerBlock = 512;
  int blocksPerGrid = (narray1 + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  double iStart = cpuSecond();
  DistGPU<<<blocksPerGrid, threadsPerBlock>>>(d_array1, d_array2, narray1,
                                              narray2, d_output);
  double iElaps = cpuSecond() - iStart;

  err = cudaMemcpy(outputGPU, d_output, narray1 * narray2 * sizeof(mt),
                   cudaMemcpyDeviceToHost);

  printf("Total computation time is %lf \n", iElaps);

  DistanceCPU(array1, array2, narray1, narray2, outputCPU);

  mt error = 0;
  bool bitequal = true;
  for (long i = 0; i < narray1 * narray2; i++) {
    error += abs(outputCPU[i] - outputGPU[i]);
    if (outputCPU[i] != outputGPU[i]) {
      bitequal = false;
    }
  }
  error /= (narray2 * narray1);

  for (int i = 0; i < 20; i++) {
    printf("CPU result %f \n", outputCPU[i]);
    printf("GPU result %f \n", outputGPU[i]);
  }

  printf("Error is %f\n", error);
  printf("Are the results bitequal? %s\n", bitequal ? "Yes" : "No");
  delete[] array1;
  delete[] array2;
  delete[] outputCPU;
  delete[] outputGPU;
  return 0;
}
