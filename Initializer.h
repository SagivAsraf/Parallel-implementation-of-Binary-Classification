#include "MainApp.h"


void initialWeightsVector(double* weightsVector, int weightsVectorSize);
void mpiInit(int argc, char *argv[], int *pId, int *numProcs);
void setInitialArgsToArgsStruct(struct initialArguments* initialArgs, int N, int k, int limit, double qc);
void structMpiInit(struct initialArguments initialArgs, MPI_Datatype* argumentsMPIType);