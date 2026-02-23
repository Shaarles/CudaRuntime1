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
/*
__global__ void example_syncthreads_dicho(int* input_data, int* output_data) {
	__shared__ int shared_data[256]; // Assuming blockDim.x = 256
	//every thread loads its data into shared memory
	shared_data[threadIdx.x] = input_data[threadIdx.x];

	__syncthreads(); // Ensure all threads have loaded their data

	// Now all threads can safely read from shared_data
	//A single thread can perform some operation on the shared data
	int n = shared_data.length; 
	int sum = 0;
	while (n < 1) {

		n /= 2;
	}
}
*/

int main() {

	dim3 blockSize(256);
	dim3 gridSize(1); // Assuming we only need one block for this example


	//initialize data on the device
	int* dev_input_data;
	int* dev_output_data;

	//initialize data on host
	int* output_data = (int*)malloc(gridSize.x * sizeof(int));

	// allocate device memory
	cudaMalloc(&dev_input_data, blockSize.x * sizeof(int));
	cudaMalloc(&dev_output_data, gridSize.x * sizeof(int));

	// Initialize input data on the host and copy to device
	initArray(dev_output_data, blockSize.x); // Assuming this function initializes the input data

	example_syncthreads << <gridSize, blockSize >> > (dev_input_data, dev_output_data);
	
	//copying into host memory
	cudaMemcpy(output_data, dev_output_data,gridSize.x * sizeof(int), cudaMemcpyDeviceToHost);

	//printing output array
	printArray(output_data, blockSize.x);

	// Free resources
	cudaFree(dev_input_data);
	cudaFree(dev_output_data);

	return 0;
}