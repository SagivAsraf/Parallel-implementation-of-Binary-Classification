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

void main(int argc, char *argv[])
{
	
	MPI_Datatype argumentsMPIType;

	struct initialArguments initialArgs;

	int numProcs, pId;
	int N, k, limit;
	int weightsVectorSize, position;

	double q, qc, alphaMax, alpha = 0, alpha0, t1;
	double minAlpha = DBL_MAX;
	double* weightsVector;
	double** points = NULL;

	bool isFoundClassification = false;
	char* results;

	mpiInit(argc, argv, &pId, &numProcs);

	/*Check if the user set a parallel program (2 or more processes)*/
	isParallelProgram(numProcs);

	/*The master read all the data from a file (the points and the parameters alpha,N,K and etc...)*/
	if (pId == MASTER) {
		/*Check the time before the execution (and BEFORE reading the file!)*/
		t1 = MPI_Wtime();

		points = readAFile(&N, &k, &alpha0, &alphaMax, &limit, &qc);
		
		/*Check if we have valid num of processes (less or equals to the number of alpha's to share between the processes)*/
		isValidNumOfProcs(numProcs, alpha0, alphaMax);
		
		setInitialArgsToArgsStruct(&initialArgs, N, k, limit, qc);
	}

	structMpiInit(initialArgs, &argumentsMPIType);

	/*Send (BroadCast) the initial arguments to all the processes */
	MPI_Bcast(&initialArgs, 1, argumentsMPIType, MASTER, MPI_COMM_WORLD);

	weightsVectorSize = initialArgs.k + 1;

	int sizeOfResultsPack = (weightsVectorSize + 2) * BYTES_IN_DOUBLE; /* 2 = alpha and q (they defined as a double)
	sorage size of a char is 1 byte , while storage size of a double is 8 bytes.*/

	results = (char*)malloc((sizeOfResultsPack) * sizeof(char));

	/*Allocate memory to the points array for each slave*/
	if (pId != MASTER) {
		points = allocatePointsArray(initialArgs.N, initialArgs.k + 2);
	}

	/*Send (BroadCast) the points array to all the processes */
	MPI_Bcast(&(points[0][0]), (initialArgs.N*((initialArgs.k) + 2)), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	/*Allocates weights Vector to each process*/
	weightsVector = (double*)malloc((weightsVectorSize) * sizeof(double));

	if (pId == MASTER) {
		sendFirstsMissionsToSlaves(numProcs, &alpha, alpha0, alphaMax);
	}
	else {
		slaveRoutine(alpha, initialArgs, pId, points, weightsVector, weightsVectorSize, &isFoundClassification, &q, results, sizeOfResultsPack, &position);
	}

	/*The Master works*/
	do {
		if (pId != MASTER) {
			break;
		}
		else {
			masterRoutine(results, sizeOfResultsPack, &position, &alpha, &q, weightsVector, weightsVectorSize, &isFoundClassification, alpha0, &minAlpha, numProcs, alphaMax);
			if (alpha >= alphaMax) {
				sendTerminationStatementToTheSlaves(numProcs);
			}
		}
	} while (alpha < alphaMax && !isFoundClassification);

	/*Create and print the results to an output file*/
	if (pId == MASTER) {
		createReports(pId, t1, isFoundClassification, q, alpha, weightsVector, weightsVectorSize, minAlpha);
	}
	else {
		MPI_Finalize();
	}
	
	freeAllTheAlocatedArrays(&results, &weightsVector,&points);
}


/******************************************* Methods Area *******************************************/

void freeAllTheAlocatedArrays(char** results, double** weightsVector, double*** points) {
	free(*results);
	free(*weightsVector);
	free(*points);
}

void masterRoutine(char* results, int sizeOfResultsPack,
	int* position, double* alpha, double* q,
	double* weightsVector, int weightsVectorSize, bool* isFoundClassification, double alpha0, double* minAlpha, int numProcs,double alphaMax) {

	MPI_Status status;

	MPI_Recv(results, sizeOfResultsPack, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	if (isTheWorkDone(status)) {


		/*UnPack the results in and add it to the results buffer (dynamic array)*/
		packOrUnPackResults(results, sizeOfResultsPack, position, alpha, q, weightsVector, weightsVectorSize, UNPACK, true);
	
		*minAlpha = *alpha;
		*isFoundClassification = true;

		/*Send to the slaves that to work is done.*/
		sendTerminationStatementToTheSlaves(numProcs);

		/*Wait for all the other slaves results*/
		collectResultsFromOthersSlaves(numProcs, results, sizeOfResultsPack, status, position, *alpha, minAlpha, *q, weightsVector, weightsVectorSize);
	
	}
	else {
		if (*alpha >= alphaMax) {
			sendTerminationStatementToTheSlaves(numProcs);
		}
		else {
			sendNewMissionToSlave(alpha, alpha0, status.MPI_SOURCE,alphaMax);
		}
	}

}

void slaveRoutine(double alpha, initialArguments initialArgs, int pId, double** points, double* weightsVector
	, int weightsVectorSize, bool* isFoundClassification, double* q, char* results, int sizeOfResultsPack, int* position) {

	MPI_Status status;

	while (true)
	{

		MPI_Recv(&alpha, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		/*If we got finish tag , we want to exit from the infinity loop.*/
		if (isTheWorkDone(status)) {
			break;
		}

		else {

			initialWeightsVector(weightsVector, weightsVectorSize);
			executeBinaryClassification(initialArgs, pId, points, weightsVector, weightsVectorSize, &alpha, isFoundClassification, q);

			/*pack the results in the results (dynamic array)*/
			packOrUnPackResults(results, sizeOfResultsPack, position, &alpha, q, weightsVector, weightsVectorSize, PACK, true);

			if (*isFoundClassification) {
				MPI_Send(results, *position, MPI_PACKED, MASTER, FINISH_TAG, MPI_COMM_WORLD);
			}
			else {
				MPI_Send(results, *position, MPI_PACKED, MASTER, WORK_TAG, MPI_COMM_WORLD);
			}
		}
	}
}


void executeBinaryClassification(initialArguments initialArgs, int pId, double** points, double* weightsVector, int weightsVectorSize, double* alpha, bool* isFoundClassification, double* q) {

	bool perfectMatch;

	for (int numOfiterations = 0; numOfiterations < initialArgs.limit; numOfiterations++)
	{
		perfectMatch = true;

		/*Iterates over the points array*/
		for (int i = 0; i < initialArgs.N ; i++)
		{
			int actualSign = (int)points[i][weightsVectorSize];
			int expectedSign = signCalculationViaWeightsVector(weightsVector, points[i], weightsVectorSize);

			if (actualSign != expectedSign)
			{
				redefinitionWeightVector(weightsVector, points[i], actualSign, expectedSign, *alpha, weightsVectorSize);
				perfectMatch = false;
				break;
			}
		}
		if (perfectMatch) {
			*q = 0;
			break;
		}
	}

	if (perfectMatch) {
		*isFoundClassification = true;
	}

	else {

		int miss = findWronglyClassifiedPointsAmountWithCuda(initialArgs.N, points, weightsVector, weightsVectorSize);

		*q = (double)(miss) / (double)initialArgs.N;
		if (*q < initialArgs.qc)
		{
			*isFoundClassification = true;
		}
	}

}

void collectResultsFromOthersSlaves(int numOfProcs, char* results, int sizeOfResultsPack, MPI_Status status, int* position, double alpha, double* minAlpha, double q, double* weightsVector, int weightsVectorSize) {

	/*numOfProcs - 2 -> we don't need to collect from the master and from the slave that already found a solution*/
	for (int i = 0; i < numOfProcs - 2; i++) {

		MPI_Recv(results, sizeOfResultsPack, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if (status.MPI_TAG == FINISH_TAG) {
			*position = 0;
			MPI_Unpack(results, sizeOfResultsPack, position, &alpha, 1, MPI_DOUBLE, MPI_COMM_WORLD);

			if (alpha < *minAlpha) {
				packOrUnPackResults(results, sizeOfResultsPack, position, &alpha, &q, weightsVector, weightsVectorSize, UNPACK, false);
				*minAlpha = alpha;
			}
		}
	}
}

void packOrUnPackResults(char* results, int sizeOfResultsPack, int* position, double* alpha, double* q, double*  weightsVector, int weightsVectorSize, int packOrUnpack, bool unPackAlpha) {

	if (packOrUnpack == PACK) {
		*position = 0;
		MPI_Pack(alpha, 1, MPI_DOUBLE, results, sizeOfResultsPack, position, MPI_COMM_WORLD);
		MPI_Pack(q, 1, MPI_DOUBLE, results, sizeOfResultsPack, position, MPI_COMM_WORLD);
		MPI_Pack(weightsVector, weightsVectorSize, MPI_DOUBLE, results, sizeOfResultsPack, position, MPI_COMM_WORLD);
	}
	else {
		if (unPackAlpha) {
			*position = 0;
			MPI_Unpack(results, sizeOfResultsPack, position, alpha, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		}

		MPI_Unpack(results, sizeOfResultsPack, position, q, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Unpack(results, sizeOfResultsPack, position, weightsVector, weightsVectorSize, MPI_DOUBLE, MPI_COMM_WORLD);
	}
}

void sendTerminationStatementToTheSlaves(int numOfProcs) {
	for (int i = 1; i < numOfProcs; i++)
	{
		MPI_Send(&i, 1, MPI_INT, i, FINISH_TAG, MPI_COMM_WORLD);
	}
}

/*	Use cuda to calculate all the expected signs for all the points, put the results in expectedSignsArray ,
over this array and calculates the number of misses points	*/
int findWronglyClassifiedPointsAmountWithCuda(int numOfPoints, double** points, double* weightsVector, int weightsVectorSize) {

	double* points_1d = (double*)malloc(sizeof(double)  * (numOfPoints * weightsVectorSize));

	combinedThe2dArrayInto1dArray(points_1d,numOfPoints,weightsVectorSize,points);

	int* expectedSignsArray = (int*)calloc(numOfPoints,sizeof(int));

	//Calculates ALL the expected signs for all the points array via the weights vector in parallel using Cuda.
	cudaError_t cudaStatus = signCalculationWithCuda(expectedSignsArray, points_1d,numOfPoints,weightsVector,weightsVectorSize);
	if (cudaStatus != cudaSuccess) {
		printf("\n(Out) -> signCalculationWithCuda failed!\n\n");
		return -1;
	}

	int miss = passOverPointsAndCountMisses(numOfPoints, points, expectedSignsArray, weightsVectorSize);

	free(points_1d);
	free(expectedSignsArray);
	return miss;

}

/*Iterates over the points array, count the misses points*/
int passOverPointsAndCountMisses(int numberOfPoints, double** points, int* expectedSignsArray, int weightsVectorSize) {

	int miss = 0;
#pragma omp parallel for reduction (+: miss) 
	for (int i = 0; i < numberOfPoints; i++) {
		int actualSign = (int)points[i][weightsVectorSize];
		int expectedSign = expectedSignsArray[i];
		if (actualSign != expectedSign)
		{
			miss++;
		}
	}

	return miss;
}

/*
	Combined the 2d array into 1d array.
	It's more accepted to send cuda 1d array and not 2d array.
*/
void combinedThe2dArrayInto1dArray(double* points_1d, int numOfPoints, int weightsVectorSize, double** points) {
#pragma omp parallel for
	for (int row = 0; row < numOfPoints; row++) {
		for (int col = 0; col < weightsVectorSize; col++) {
			points_1d[(row * weightsVectorSize) + col] = points[row][col];
		}
	}
}

void sendNewMissionToSlave(double* alpha, double alpha0, int slaveId,double alphaMax) {
	incrementAlphaByAlpha0(alpha, alpha0);
	if (*alpha > alphaMax) {
		return;
	}
	MPI_Send(alpha, 1, MPI_DOUBLE, slaveId, WORK_TAG, MPI_COMM_WORLD);
}

void sendFirstsMissionsToSlaves(int numofproc, double* alpha, double alpha0, double alphaMax) {
	for (int i = 1; i < numofproc; i++)
	{
		incrementAlphaByAlpha0(alpha, alpha0);
		if (*alpha > alphaMax) {
			break;
		}
		MPI_Send(alpha, 1, MPI_DOUBLE, i, WORK_TAG, MPI_COMM_WORLD);
	}
}

int signCalculationViaWeightsVector(double* weightsVector, double* point, int weightsVectorSize)
{
	/*Each point has a weight, we use the next formula for calculate the sign of the mulpilicity between the weights and the point's coordiantes.*/
	double sum = 0;
	int i;

	for (i = 0; i < weightsVectorSize; i++)
	{
		sum += weightsVector[i] * point[i];
	}

	return sum >= 0 ? 1 : -1;
}

void redefinitionWeightVector(double* weightsVector, double* point, int actualSign, int expectedSign, double alpha, int weightsVectorSize)
{
	for (int i = 0; i < weightsVectorSize; i++)
	{
		weightsVector[i] += (alpha * ((actualSign - expectedSign)/2)) * point[i];
	}
}

void incrementAlphaByAlpha0(double* alpha, double alpha0)
{
	*alpha = *alpha + alpha0;
}

