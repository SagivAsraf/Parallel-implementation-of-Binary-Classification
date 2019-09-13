#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include <math.h>
#include <float.h>

#include "Utils.h"
#include "Initializer.h"
#include "HandleFiles.h"


#define MASTER 0
#define ARGS_IN_STRUCT 4
#define NUM_VARS_IN_RESULTS_STRUCT 3
#define WORK_TAG 1
#define FINISH_TAG 2
#define UNPACK 0
#define PACK 1
#define WAS_FOUND 1
#define WAS_NOT_FOUND 0

#define BYTES_IN_DOUBLE 8

struct initialArguments
{
	int N;
	int k;
	int limit;
	double qc;
};


cudaError_t signCalculationWithCuda(int* expectedSignsArray,double* points, int numOfPoints, double* weightsVector, int weightsVectorSize);

void combinedThe2dArrayInto1dArray(double* points_1d, int numOfPoints, int weightsVectorSize, double** points);

int passOverPointsAndCountMisses(int numberOfPoints, double** points, int* expectedSignsArray, int weightsVectorSize);

void freeAllTheAlocatedArrays(char** results, double** weightsVector, double*** poin);

int findWronglyClassifiedPointsAmountWithCuda(int numOfPoints, double** points, double* weightsVector, int weightsVectorSize);


int signCalculationViaWeightsVector(double* weightsVector, double* point, int weightsVectorSize);
void redefinitionWeightVector(double* weightsVector, double* point, int actualSign, int expectedSign, double alpha, int weightsVectorSize);
void incrementAlphaByAlpha0(double* alpha, double alpha0);


void sendFirstsMissionsToSlaves(int numofproc, double* alpha, double alpha0, double alphaMax);

void executeBinaryClassification(initialArguments initialArgs, int pId, double** points, double* weightsVector, int weightsVectorSize, double* alpha, bool* isFoundClassification, double* q);
void sendNewMissionToSlave(double* alpha, double alpha0, int slaveId, double alphaMax);
void sendTerminationStatementToTheSlaves(int numOfProcs);
void packOrUnPackResults(char* results, int sizeOfResultsPack, int* position, double* alpha, double* q, double*  weightsVector, int weightsVectorSize, int packOrUnpack, bool unPackAlpha);

void masterRoutine(char* results, int sizeOfResultsPack,
	int* position, double* alpha, double* q,
	double* weightsVector, int weightsVectorSize, bool* isFoundClassification, double alpha0, double* minAlpha, int numProcs, double alphaMax);

void slaveRoutine(double alpha, initialArguments initialArgs, int pId, double** points, double* weightsVector
	, int weightsVectorSize, bool* isFoundClassification, double* q, char* results, int sizeOfResultsPack, int* position);


void collectResultsFromOthersSlaves(int numOfProcs, char* results, int sizeOfResultsPack, MPI_Status status, int* position, double alpha, double* minAlpha, double q, double* weightsVector, int weightsVectorSize);