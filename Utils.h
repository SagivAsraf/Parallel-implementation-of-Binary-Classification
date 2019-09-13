#include "MainApp.h"


bool isTheWorkDone(MPI_Status status);
void isParallelProgram(int numprocs);

void isValidNumOfProcs(int numprocs,double alpha0, double alphaMax);

void printWeightsVector(double* weightsVector, int weightsVectorSize);
void printFirstLineParametersData(int N, int k, double alpha0, double alphaMax, int limit, double qc);
double** allocatePointsArray(int numOfPoints, int k);
void createReports(int pId, double t1, bool isFoundClassification, double q, double alpha, double* weightsVector, int weightsVectorSize, double minAlpha);
