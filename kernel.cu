#include "Vector.cpp"
#include "kernel.h"


__global__ void add(int* a, int*b, int* c, int vector_length ){
	//Make computations
	int work_index = threadIdx.x * threadIdx.y;
	// It is possible to verify with a "work index" that you are not getting out of bound if you put too many threads to work

	if (work_index < vector_length) {
		a[threadIdx.x * threadIdx.y] = blockIdx.x;
		b[threadIdx.x * threadIdx.y] = blockIdx.y;
		c[threadIdx.x * threadIdx.y] = blockIdx.x + blockIdx.y;
	}
}
/*
void unifiedMemExample(int vectorLength) {

	//Pointers to memory vectors
	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;

	//Using unified memory to allocate buffers
	cudaMallocManaged(&A, vectorLength * sizeof(float));
	cudaMallocManaged(&B, vectorLength * sizeof(float));
	cudaMallocManaged(&C, vectorLength * sizeof(float));

	//Initialize vectors A and B on the host
	initArray(A, vectorLength);
	initArray(B, vectorLength);
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}
*/
int main() {
	dim3 grid(4, 4);
	int gridsize = grid.x * grid.y;
	dim3 block(16, 16);
	int blockSize = block.x * block.y;

	int* a = (int*)malloc(blockSize * sizeof(int));
	int* b = (int*)malloc(blockSize * sizeof(int));
	int* c = (int*)malloc(blockSize * sizeof(int));
	add <<<grid, block >>> (a, b, c, blockSize);

	for (int i = 0; i < blockSize; i++) {
		cout << "a[" << i << "] = " << a[i] << endl;
		cout << "b[" << i << "] = " << b[i] << endl;
		cout << "c[" << i << "] = " << c[i] << endl;
	}
	
	free(a);
	free(b);
	free(c);

	return 0;
}