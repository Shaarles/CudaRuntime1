#include "SIMTtuto.h"

__global__ void example_syncthreads(int* input_data, int* output_data) {
	__shared__ int shared_data[256]; // Assuming blockDim.x = 256
	//static allocation of shared memory, the size must be known at compile time

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

__global__ void example_syncthreads_dynamic_allocation(int* input_data, int* output_data) {
	extern __shared__ int shared_data[]; // Assuming blockDim.x = 256
	//dynamic allocation of shared memory, the size must be known at compile time
	//size in byte given in the triple chevroton notation on kernel launch

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

__global__ void example_syncthreads_dicho(int* input_data, int* output_data) {
	__shared__ int shared_data[256]; // Assuming blockDim.x = 256
	//every thread loads its data into shared memory
	shared_data[threadIdx.x] = input_data[threadIdx.x];

	__syncthreads(); // Ensure all threads have loaded their data

	// Now all threads can safely read from shared_data
	//A single thread can perform some operation on the shared data
	int n = 256; 
	int sum = 0;
	while (n < 1) {

		
	}
}


int main() {

	dim3 blockSize(256);
	dim3 gridSize(1); // Assuming we only need one block for this example


	//initialize data on the device
	int* dev_input_data;
	int* dev_output_data;


	/*
	* not a really good option 
	//initialize data on host
	int* input_data = (int*)malloc(gridSize.x * sizeof(int));
	int* output_data = (int*)malloc(gridSize.x * sizeof(int));
	*/

	//better option

	int* input_data = nullptr;
	int* output_data = nullptr;

	cudaMallocHost(&input_data, blockSize.x * sizeof(int));
	cudaMallocHost(&output_data, blockSize.x * sizeof(int));


	// allocate device memory
	cudaMalloc(&dev_input_data, blockSize.x * sizeof(int));
	cudaMalloc(&dev_output_data, gridSize.x * sizeof(int));

	// Initialize input data on the host and copy to device
	initArray(input_data, blockSize.x); // Assuming this function initializes the input data

	//copying into device memory
	cudaMemcpy(dev_input_data, input_data, blockSize.x * sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel with static shared memory allocation
	/*
	example_syncthreads << <gridSize, blockSize >> > (dev_input_data, dev_output_data);
	*/

	// Launch kernel with dynamic shared memory allocation
	example_syncthreads_dynamic_allocation << <gridSize, blockSize, 256 * sizeof(int) >> > (dev_input_data, dev_output_data);

	//copying into host memory
	cudaMemcpy(output_data, dev_output_data,gridSize.x * sizeof(int), cudaMemcpyDeviceToHost);

	//printing output array
	printArray(output_data, gridSize.x);
	// Free resources
	cudaFree(dev_input_data);
	cudaFree(dev_output_data);
	free(output_data);
	free(input_data);

	return 0;
}