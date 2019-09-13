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

void setInitialArgsToArgsStruct(struct initialArguments* initialArgs, int N, int k, int limit, double qc) {

	initialArgs->N = N;
	initialArgs->k = k;
	initialArgs->limit = limit;
	initialArgs->qc = qc;
}

void structMpiInit(struct initialArguments initialArgs, MPI_Datatype* argumentsMPIType) {

	MPI_Datatype type[ARGS_IN_STRUCT] = { MPI_INT, MPI_INT,MPI_INT,MPI_DOUBLE };
	int blocklen[ARGS_IN_STRUCT] = { 1, 1, 1, 1 };
	MPI_Aint disp[ARGS_IN_STRUCT];

	// Create MPI user data type for initialArgs
	disp[0] = (char *)&initialArgs.N - (char *)&initialArgs;
	disp[1] = (char *)&initialArgs.k - (char *)&initialArgs;
	disp[2] = (char *)&initialArgs.limit - (char *)&initialArgs;
	disp[3] = (char *)&initialArgs.qc - (char *)&initialArgs;

	MPI_Type_create_struct(ARGS_IN_STRUCT, blocklen, disp, type, argumentsMPIType);
	MPI_Type_commit(argumentsMPIType);
}

void mpiInit(int argc, char *argv[], int *pId, int *numProcs)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, pId);
	MPI_Comm_size(MPI_COMM_WORLD, numProcs);
}

void initialWeightsVector(double* weightsVector, int weightsVectorSize)
{
	#pragma omp parallel for shared(weightsVector)
	for (int i = 0; i < weightsVectorSize; i++)
	{
		weightsVector[i] = 0;
	}
}
