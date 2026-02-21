#include "kernel.h"


__global__ void add(float* a, float*b, float* c, int vector_length ){
	//Make computations
	int work_index = blockDim.x*blockIdx.x+ threadIdx.x;
	// It is possible to verify with a "work index" that you are not getting out of bound if you put too many threads to work

	if (work_index < vector_length) {
		c[work_index] = a[work_index] + b[work_index];
	}
}

void add_cpu_vers(float* a, float* b, float* c, int vector_length) {
	//Make computations
	for (int i=0; i < vector_length; i++) {
		c[i] = a[i] + b[i];
	}
}


bool compareResults(float* a, float* b, int length) {
	for (int i = 0; i < length; i++) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

void unifiedMemExample(int vectorLength) {
	//defining add parameters

	int threads = 256;
	int blocks = (vectorLength + threads - 1) / threads; //Ceil of vectorLength/threads

	//Pointers to memory vectors
	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;
	float* comparisonResult = (float*)malloc(vectorLength * sizeof(float));

	//Using unified memory to allocate buffers
	cudaMallocManaged(&A, vectorLength * sizeof(float));
	cudaMallocManaged(&B, vectorLength * sizeof(float));
	cudaMallocManaged(&C, vectorLength * sizeof(float));

	initArray(A, vectorLength);
	initArray(B, vectorLength);
	//Measuring time taken to do the operation on the GPU
	auto start = chrono::high_resolution_clock::now();
	//Launch kernel on the GPU
	add << <blocks, threads >> > (A, B, C, vectorLength);
	auto end = chrono::high_resolution_clock::now();
	cout << "GPU version took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;


	//important part incoming!!!!
	cudaDeviceSynchronize();	//That's so important!!!!

	//Measuring time taken to do the operation on the CPU
	start = chrono::high_resolution_clock::now();
	add_cpu_vers(A, B, comparisonResult, vectorLength);
	end = chrono::high_resolution_clock::now();
	cout << "CPU version took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;
	
	cout << compareResults(C, comparisonResult, vectorLength) << endl;


	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(comparisonResult);
}

void explicitMemExample(int vectorLength) {
	//defining add parameters
	int threads = 256;
	int blocks = (vectorLength + threads - 1) / threads; //Ceil of vectorLength/threads

	//Pointers for host memoy
	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;
	float* comparison = nullptr;

	//Pointers for device memoy
	float* devA = nullptr;
	float* devB = nullptr;
	float* devC = nullptr;

	//Allocate Host Memory using cudaMallocHost API
	//(best option when buffers will be use for copies between CPU and GPU mem)
	//(for best performance, use pinned memory for buffers that will be used for copies between CPU and GPU memory)
	cudaMallocHost(&A, vectorLength * sizeof(float));
	cudaMallocHost(&B, vectorLength * sizeof(float));
	cudaMallocHost(&C, vectorLength * sizeof(float));
	cudaMallocHost(&comparison, vectorLength * sizeof(float));

	//Allocating device memory
	initArray(A, vectorLength);
	initArray(B, vectorLength);

	//start allocating and copying memory on the gpu
	//Allocating memory to GPU
	cudaMalloc(&devA, vectorLength * sizeof(float));
	cudaMalloc(&devB, vectorLength * sizeof(float));
	cudaMalloc(&devC, vectorLength * sizeof(float));

	//Copy data to the GPU
	cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devC, C, vectorLength * sizeof(float), cudaMemcpyHostToDevice);

	//Measuring time taken to do the operation on the GPU
	auto start = chrono::high_resolution_clock::now();
	//Launch kernel on the GPU
	add << <blocks, threads >> > (devA, devB, devC, vectorLength);
	auto end = chrono::high_resolution_clock::now();
	cout << "GPU version took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;


	//important part incoming!!!!
	cudaDeviceSynchronize();	//That's so important!!!!

	//Measuring time taken to do the operation on the CPU
	start = chrono::high_resolution_clock::now();
	add_cpu_vers(A, B, C, vectorLength);
	end = chrono::high_resolution_clock::now();
	cout << "CPU version took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;

	cudaMemcpy(comparison, devC, vectorLength * sizeof(float), cudaMemcpyDefault);

	cout << compareResults(C, comparison, vectorLength) << endl;
	//Freeing all this memory used
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
}

int main() {
	char a;
	int n = 20000000;
	explicitMemExample(50*n);
	cin >> a;

	return 0;
}