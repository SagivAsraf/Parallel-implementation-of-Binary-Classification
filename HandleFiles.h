#include "MainApp.h"

double** readAFile(int* N, int* k, double* alpha0, double* alphaMax, int* limit, double* qc);
void readFirstLineParametersData(FILE* file, int* N, int* k, double* alpha0, double* alphaMax, int* limit, double* qc);
double** readAllPointsFromFile(FILE* file, int N, int k);
void checkIfFileExist(FILE* file);
void printResultsReport(double alpha, double q, double* weightsVector, int weightsVectorSize,int isFound);
void printWeightVectorToTheFile(FILE* file, double* weightsVector, int weightsVectorSize);