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

#include "HandleFiles.h"

#define _CRT_SECURE_NO_WARNINGS
#define ERROR_CODE 0

double** readAFile(int* N, int* k, double* alpha0, double* alphaMax, int* limit, double* qc) {

	FILE *file;

	char fileFullPath[] = "C:\\Sagiv\\data1.txt";

	file = fopen(fileFullPath, "r");

	checkIfFileExist(file);

	readFirstLineParametersData(file, N, k, alpha0, alphaMax, limit, qc);

	double** points = readAllPointsFromFile(file, *N, *k + 2);

	fclose(file);

	return points;
}



void readFirstLineParametersData(FILE* file, int* N, int* k, double* alpha0, double* alphaMax, int* limit, double* qc) {
	
	/*Reading all the parameters from the first line of the file*/
	fscanf(file, "%d", N);
	if (*N <= 0) {
		printf("Invalid input. please insert a positive number of points. \n");
		exit(ERROR_CODE);
	}

	fscanf(file, "%d", k);
	fscanf(file, "%lf", alpha0);
	fscanf(file, "%lf", alphaMax);
	fscanf(file, "%d", limit);
	fscanf(file, "%lf", qc);

}

double** readAllPointsFromFile(FILE* file, int N, int k) {
	
	double** points = allocatePointsArray(N, k);

	int i;
	int lastPosition = k - 2;
	
	for (i = 0; i < N && !feof(file); i++) {
		points[i][lastPosition] = 1;
		for (int j = 0; j < k; j++)
		{
			if (j != lastPosition) {
				fscanf(file, "%lf", &(points[i][j]));
			}
		}
	}
	return points;
}

void printResultsReport(double alpha, double q, double* weightsVector, int weightsVectorSize, int isFound)
{

	char fileFullPath[] = "C:\\Users\\cudauser\\Desktop\\312527450_Sagiv_Asraf_Parallel_Development_Final_Project\\FinalProjectParallelSagivAsraf\\output.txt";

	/*Please enter a valid path (project directory path)*/
	FILE *file = fopen(fileFullPath, "w");

	checkIfFileExist(file);

	if (isFound == WAS_FOUND) {
		fprintf(file, "Alpha Minimum = %lf , Q = %lf \n", alpha, q);
		printWeightVectorToTheFile(file, weightsVector, weightsVectorSize);
	}
	else {
		fprintf(file, "Alpha was not found");
	}

	fclose(file);


}

void printWeightVectorToTheFile(FILE* file, double* weightsVector, int weightsVectorSize) {
	for (int i = 0; i < weightsVectorSize; i++)
	{
		fprintf(file, "W[%d] = %lf\n", i, weightsVector[i]);
	}
}

void checkIfFileExist(FILE* file) {
	if (!file)
	{
		printf("ERROR via opening file -> terminating the program. \n");
		exit(ERROR_CODE);
	}
}