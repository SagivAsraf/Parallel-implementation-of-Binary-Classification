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


#include "Utils.h"

void createReports(int pId, double t1, bool isFoundClassification, double q, double alpha, double* weightsVector, int weightsVectorSize, double minAlpha) {

	/*Check the time AFTER the execution*/

	double t2 = MPI_Wtime();
	printf("------> Total time : %f \n", t2 - t1);

	MPI_Finalize();

	if (isFoundClassification) {
		printf("*******>>> \n PID (MASTER) : %d Q is = %lf MINALPHA FINALY: %lf\n*******>>>\n\n", pId, q, minAlpha);
		fflush(stdout);
		printWeightsVector(weightsVector, weightsVectorSize);
		printResultsReport(minAlpha, q, weightsVector, weightsVectorSize, WAS_FOUND);

	}
	else {
		printf("\nClassification was not FOUND\n");
		fflush(stdout);
		printResultsReport(0, 0, NULL, 0, WAS_NOT_FOUND);
	}
}


void printWeightsVector(double* weightsVector, int weightsVectorSize)
{
	for (int i = 0; i < weightsVectorSize; i++)
	{
		printf("W[%d] = %lf \n", i, weightsVector[i]);
	}
}

void printFirstLineParametersData(int N, int k, double alpha0, double alphaMax, int limit, double qc) {
	printf("N: %d\n", N);
	printf("k: %d\n", k);
	printf("alpha0: %f\n", alpha0);
	printf("alphaMax: %f\n", alphaMax);
	printf("limit: %d\n", limit);
	printf("qc: %f\n", qc);
}

void isParallelProgram(int numprocs) {
	if (numprocs <= 1) {
		printf("Please set more processes for a paraller compution");
		MPI_Abort(MPI_COMM_WORLD, 999);
	}
}

void isValidNumOfProcs(int numprocs, double alpha0, double alphaMax) {
	int amountOfAlphas = (int)floor(alphaMax / alpha0);
	if (numprocs > amountOfAlphas) {
		printf("\n***\nThere are just %d alphas to share between the processes, please set maximum %d processes (or increase alpha max)\n***",amountOfAlphas,amountOfAlphas );
		MPI_Abort(MPI_COMM_WORLD, 999);
	}
}

bool isTheWorkDone(MPI_Status status) {
	return status.MPI_TAG == FINISH_TAG;
}

/*
Helper method for allocating an 2d array of points.
**** Taken from stackOverflow forum: "questions/5104847/mpi-bcast-a-dynamic-2d-array"*/

double** allocatePointsArray(int numOfPoints, int k) {

	/* allocate the numOfPoints * k contiguous items */
	double *point = (double *)malloc(numOfPoints * k * sizeof(double));

	/* allocate the row pointers into the memory */
	double** pointsArr = (double **)malloc(numOfPoints * sizeof(double*));

	/* set up the pointers into the contiguous memory */
	for (int i = 0; i < numOfPoints; i++)
		pointsArr[i] = &(point[k*i]);

	return pointsArr;
}