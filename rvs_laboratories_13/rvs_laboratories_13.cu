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

__global__ void addBlockSums(float *output, float *partialSums, int len) {
  int tID = threadIdx.x;
  int bID = blockIdx.x;
  int start = 2 * bID * blockDim.x;
  int offset = 2 * tID;

  float sum = (bID > 0) ? partialSums[bID - 1] : 0.0f;

  if (start + offset < len) {
    output[start + offset] += sum;
  }
  if (start + offset + 1 < len) {
    output[start + offset + 1] += sum;
  }
}

__global__ void scan(float *input, float *output, float *partialSums, int len) {
  extern __shared__ float temp[];

  int tID = threadIdx.x;
  int bID = blockIdx.x;
  int start = 2 * bID * blockDim.x;
  int offset = 2 * tID;

  temp[offset] = (start + offset < len) ? input[start + offset] : 0.0f;
  temp[offset + 1] = (start + offset + 1 < len) ? input[start + offset + 1] : 0.0f;

  __syncthreads();

  for (int d = 1; d <= blockDim.x; d *= 2) {
    int index = 2 * d * (tID + 1) - 1;
    if (index < 2 * blockDim.x) {
      temp[index] += temp[index - d];
    }
    __syncthreads();
  }

  if (partialSums != NULL) {
    if (tID == 0) {
      partialSums[bID] = temp[2 * blockDim.x - 1];
    }
  }

  if (tID == 0) {
    temp[2 * blockDim.x - 1] = 0.0f;
  }
  __syncthreads();

  for (int d = blockDim.x; d >= 1; d /= 2) {
    int index = 2 * d * (tID + 1) - 1;
    if (index < 2 * blockDim.x) {
      float t = temp[index - d];
      temp[index - d] = temp[index];
      temp[index] += t;
    }
    __syncthreads();
  }

  if (start + offset < len) {
    output[start + offset] = temp[offset] + input[start + offset];
  }
  if (start + offset + 1 < len) {
    output[start + offset + 1] = temp[offset + 1] + input[start + offset + 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // Входной одномерный список
  float *hostOutput; // Выходной список
  float *deviceInput;
  float *deviceOutput;
  float *devicePartialSums; // Массив для хранения сумм блоков
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

  dim3 dimBlock(BLOCK_SIZE, 1);
  int numBlocks = (numElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
  dim3 dimGrid(numBlocks, 1);
  size_t sharedMemSize = 2 * BLOCK_SIZE * sizeof(float);

  wbCheck(cudaMalloc((void **)&devicePartialSums, numBlocks * sizeof(float)));

  wbTime_start(Compute, "Performing CUDA computation");
  scan<<<dimGrid, dimBlock, sharedMemSize>>>(deviceInput, deviceOutput, devicePartialSums, numElements);

  if (numBlocks > 1) {
    scan<<<dim3(1,1), dimBlock, sharedMemSize>>>(devicePartialSums, devicePartialSums, NULL, numBlocks);
    addBlockSums<<<dimGrid, dimBlock>>>(deviceOutput, devicePartialSums, numElements);
  }

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(devicePartialSums);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
