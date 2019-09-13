/**
* The Binary Classification program implements the Binary Classification algorithm in parallel development,
* using MPI,OpenMP and Cuda (By Invidia)
*
* @author Sagiv Asraf
* @Id : 312527450
* @since 25-08-2019
* Lecturer : Dr. Boris Moroz.
*
*/

#include "MainApp.h"

#define NUM_OF_THREADS_PER_BLOCK 1024

__global__ void calcSign(int numOfPoints, int *expectedSigns, double* pointsArray, double* weightsArray, int weightsVectorSize)
{

	/*CudaThread -> as index of point to check. 
	For example: 
	CudaThread: 20032 check the point 20032 from the points array and so on.
	*/

	int cudaThread = threadIdx.x + (blockIdx.x * blockDim.x);

	if (cudaThread >= numOfPoints) {
		return;
	}

	double sum = 0;

	for (int i = 0; i < weightsVectorSize; i++)
	{
		/*	Each point has a weight, we use the next formula for calculate the sign of the mulpilicity
		between the weights and the point's coordiantes.	*/
		sum += weightsArray[i] * pointsArray[(cudaThread * weightsVectorSize) + i];
	}

	sum >= 0 ? expectedSigns[cudaThread] = 1 : expectedSigns[cudaThread] = -1;

}

// Helper method for using CUDA to calcuate the sign of the points via the weights vector.
cudaError_t signCalculationWithCuda(int* expectedSignsArray, double* points, int numOfPoints, double* weightsVector, int weightsVectorSize)
{

	double* pointsArray_Cuda = 0;
	double* weightsArray_Cuda = 0;
	int* expctedArray_Cuda = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n***");
		goto Error;
	}

	/*Allocate memory for pointsArray_Cuda*/
	cudaStatus = cudaMalloc((void**)&pointsArray_Cuda, sizeof(double) * (numOfPoints * weightsVectorSize));
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaMalloc failed!\n***");
		goto Error;
	}

	//Copy input arrays from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(pointsArray_Cuda, points, sizeof(double)  * (numOfPoints * weightsVectorSize), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaMemcpy Of PointsArray failed!\n***");
		goto Error;
	}
	/*Allocate memory for weightsArray_Cuda*/
	cudaStatus = cudaMalloc((void**)&weightsArray_Cuda, weightsVectorSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaMalloc failed!\n***");
		goto Error;
	}

	/*Allocate memory for expctedArray_Cuda*/
	cudaStatus = cudaMalloc((void**)&expctedArray_Cuda, numOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaMalloc failed!\n***");
		goto Error;
	}

	//Copy input arrays from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(weightsArray_Cuda, weightsVector, weightsVectorSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaMemcpy failed!\n***");
		goto Error;
	}

	/* Nvidia formula
	https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
	*/

	int numOfBlocks = (numOfPoints + NUM_OF_THREADS_PER_BLOCK - 1) / NUM_OF_THREADS_PER_BLOCK;

	// Launch a kernel on the GPU with numOfBlocks blocks and NUM_OF_THREADS_PER_BLOCK threads per each block.
	calcSign << <numOfBlocks, NUM_OF_THREADS_PER_BLOCK >> >(numOfPoints, expctedArray_Cuda, pointsArray_Cuda, weightsArray_Cuda, weightsVectorSize);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("\n***calcSign launch failed: %s\n\n***", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Copy output array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(expectedSignsArray, expctedArray_Cuda, (numOfPoints * sizeof(int)), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("\n***cudaMemcpy LAST failed!\n\n***");
		goto Error;
	}

Error:
	cudaFree(expctedArray_Cuda);
	cudaFree(pointsArray_Cuda);
	cudaFree(weightsArray_Cuda);

	return cudaStatus;
}