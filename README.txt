1) In order to run the program, 
please enter the full path of the points file, into the array: "fileFullPath" (that exist in "readAFile" method in the HandleFile.cpp file).

For example, I have a file named "data1.txt" in my C drive, so the full path will be: 
"C:\\data1.txt"


2) Please choose the number of processes to be less or equal to the num of alpha's that the algorithm has to check.
For example : alpha0 = 0.1 , alphaMax = 0.9 , so we have 9 alphas to share , and in this case the number of processes will be maximum 9. 
(This is the way I chose to make my code to run in parallel , so each processes gets is own alpha more details later)

3) OutPut file -> please enter a valid path into the array: "fileFullPath" (that exist in ""printResultsReport" method in the HandleFile.cpp file).
we required to write the report into file named: "output.txt" in the project directory,
so if I saved the project on the desktop of the cudaUser (connecting via vmware Horizon Client in Afeka)
so the full path will be: ""

"C:\\Users\\cudauser\\Desktop\\312527450_Sagiv_Asraf_Parallel_Development_Final_Project\\FinalProjectParallelSagivAsraf\\output.txt"


*I used MPI+OpenMP+CUDA configuration.*

Code explanation:




I chose to implement the code using the dynamic division of labor attitude.


First of all, the master read all the data from the points file, and after this, I used BroadCast (MPI) to share all data between all
the processes, so each process will able to access the data and handle it.


I decided to parallelize the work in a way that each process will get alpha to check, and in this way, more than one alpha is checked by time. 


As expected from dynamic attitude, the master starts with sharing the first missions with the other processes,
and then wait for the processess to return an answer.


The processes can return two types of answer:

1) I'm DONE! I found a solution for the problem (FINISH_TAG)

2) I did not find any solution, please give me a new alpha to check (WORK_TAG)

 

In the first case, 
the master will immediately send a new mission for this process.
(in case he has more alpha to share!) 


However, in the second case, (or if he does not have any another alpha to share)
the master will send TERMINATION message for the processes, and let them know that the work is done.


The master will collect the solutions from the other processes that already did the work, 
in order to check if their solution is better than the first solution (lower alpha as requested).
I am looking for the lowest alpha , so I handle a logic of finding a minimin alpha ,and replace the results 
in case that another processes found lower alpha.

The Algorithm:



Each process iterates over the points array (limit* times)
.
Per each iteration, the process will iterate over the points array,
 
and calculate the expected sign of each point.

If all the points have the same sign as the expectations, 
we found a perfect match, (q = 0) , and this is the solution (the weightsVector does not changed since the last time it changed).


The first point that doesn't satisfies this critertion, 
will stop the check and to redifinition of the weights array.


After this, I check if I don't have a perfect match (no misses at all!) 

we want to calculate the number of misses we have.


I did it with CUDA.

I send to the cuda program points array, weights array and expected signs array so in parallel the GPU threads will check each point and the expected sign and put all the results in the expected signs array.


So, after I have the expected signs array, 
I just have to iterates over it and check how much misses I have. 

If the number of misses/number of points is lower that qc*, so the process found a solution!
If not, it's mean that no classification was found by this process.
If all the processes won't found any solution, so a propper messafe will be printing into the output file.



I used OpenMP to count the amount of misses (for reduction (+: miss)),
so 4 threads (mine default is 4 threads 4 cores of processor) 
will iterates the expected signs array and count the misses, 
in this way it will be executed faster and parallel.


I used OpenMP also for initialize the weights vector.



*limit - the maximum iterations allowed to iterates over the points array and looking for a perfect match or redefinition of the weights vector.

*qc - the Quality of Classifier 


