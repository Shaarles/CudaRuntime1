#include "SIMTtuto.h"

__global__ void example_syncthreads(int* input_data, int* output_data) {
	__shared__ int shared_data[256]; // Assuming blockDim.x = 256
	//every thread loads its data into shared memory
	shared_data[threadIdx.x] = input_data[threadIdx.x];

	__syncthreads(); // Ensure all threads have loaded their data

	// Now all threads can safely read from shared_data
	//A single thread can perform some operation on the shared data
	if (threadIdx.x == 0) {
		int sum = 0;
		for (int i = 0; i < blockDim.x; ++i) {
			sum += shared_data[i];
		}
		output_data[blockIdx.x] = sum; // Store the result for this block
	}
}	

int main() {

	dim3 blockSize(256);
	dim3 gridSize(1); // Assuming we only need one block for this example

	//initialize data on the host
	int* input_data;
	int* output_data;
	int* comparison;

	//initialize data on the device
	int* dev_input_data;
	int* dev_output_data;

	//allocate memory on the host
	cudaMallocHost(&input_data, blockSize.x * sizeof(int));
	cudaMallocHost(&output_data, blockSize.x * sizeof(int));
	cudaMallocHost(&comparison, blockSize.x * sizeof(int));

	// allocate device memory
	cudaMalloc(&dev_input_data, blockSize.x * sizeof(int));
	cudaMalloc(&dev_output_data, blockSize.x * sizeof(int));



	// Free resources
	cudaFree(dev_input_data);
	cudaFree(dev_output_data);

	cudaFreeHost(input_data);
	cudaFreeHost(output_data);

	return 0;
}