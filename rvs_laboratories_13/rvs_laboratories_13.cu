// На вход подается список (lst) длины n
// Результатом является префиксная сумма = {lst[0], lst[0] + lst[1],
// lst[0] + lst[1] + ... + lst[n-1]}

#include "wb.h"

#define BLOCK_SIZE 512 //@@ Это можно поменять

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  __shared__ float temp[BLOCK_SIZE * 2];

  int tID = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;

  int i = start + tID;
  int j = start + tID + blockDim.x;

  temp[tID] = (i < len) ? input[i] : 0;
  temp[tID + blockDim.x] = (j < len) ? input[j] : 0;

  // Upsweep (редукция)
  int offset = 1;
  for (int d = blockDim.x; d > 0; d >>= 1) {
    __syncthreads();
    if (tID < d) {
      int ai = offset * (2 * tID + 1) - 1;
      int bi = offset * (2 * tID + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }

  // Обнуляем последний элемент
  if (tID == 0) {
    temp[2 * blockDim.x - 1] = 0;
  }

  // Downsweep
  for (int d = 1; d < 2 * blockDim.x; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tID < d) {
      int ai = offset * (2 * tID + 1) - 1;
      int bi = offset * (2 * tID + 2) - 1;

      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();

  // Запись результата
  if (i < len) output[i] = temp[tID] + input[i];
  if (j < len) output[j] = temp[tID + blockDim.x] + input[j];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // Входной одномерный список
  float *hostOutput; // Выходной список
  float *deviceInput;
  float *deviceOutput;
  int numElements; // количество элементов в списке

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Инициализируйте размерности блоков и сетки
  int threads = BLOCK_SIZE;
  int blocks = (numElements + threads * 2 - 1) / (threads * 2);

  wbTime_start(Compute, "Performing CUDA computation");
  scanKernel<<<blocks, threads>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
