/* Exercise to convert a simple serial code to count the number of prime                             
   numbers within a given interval into MPI (32-bit, int version).                                   
                                                                                                     
   All prime numbers can be expressed as 6*k-1 or 6*k+1, k being an                                  
   integer. We provide the range of k to probe as macro parameters                                   
   KMIN and KMAX (see below).                                                                        
                                                                                                     
   Check the parallel code correctness - it should produce the same number of prime                  
   numbers as the serial version, for the same range KMIN...KMAX. (The result                        
   is 3,562,113 for K=1...10,000,000.)                                                               
                                                                                                     
   Try to make the parallel code as efficient as possible.                                           
                                                                                                     
   Your speedup should be close to the number of threads/ranks you are using.                        
                                                                                                     
   Describe everything you do via detailed comments.                                                 
                                                                                                     
                                                                                                     
MPI instructions and hints:                                                                          
                                                                                                     
   You can choose one of the three levels of difficulty, with Level 1 being the                      
   easiest (and giving you the lowest maximum grade, 80%). If you want the maximum (100%) grade,     
   you should choose Level 3. Any bugs/mistakes/lack of commenting etc. will further reduce          
   the grade. Please indicate the Level chosen in the header of your solution.                       
                                                                                                     
General instructions (for any level):                                                                
* You don't have to print continuosuly the globally best itinerary found so far, the way             
  it was done in the serial and OpenMP versions. Only the final solution needs to be printed,        
  at the end.                                                                                        
* No assumptions about the KMAX-KMIN+1 being integer dividable by the number of ranks should be made\
.                                                                                                    
  In other words, your code should give the correct result and be efficient with any number of ranks\
.                                                                                                    
* Reduce the number of communications to a minimum.                                                  
* It should be the rank 0 responsibility to call gettimeofday() function, and print                  
  all the messages.                                                                                  
* Place the first timer right after MPI_Init/Comm_size/Comm_rank functions; the second timer         
  should go right before rank 0 prints the final results.                                            
                                                                                                     
                                                                                                     
If you are doing Level 1 (static workload balancing; maximum grade 80%):                             
* Use simple static way of distributing the workload.                                                
* All ranks (including rank 0) should participate in computations.                                   
* Use the following functions:                                                                       
 - MPI_Init                                                                                          
 - MPI_Finalize                                                                                      
 - MPI_Comm_rank                                                                                     
 - MPI_Comm_size                                                                                     
 - MPI_Reduce                                                                                        
                                                                                                     
                                                                                                     
If you are doing Level 2 (dynamic workload balancing with the idle master; maximum grade 90%):
* Going from KMAX to KMIN will facilitate dynamic workload balancing.                                
* Introduce a chunk parameter dK (can be a macro parameter: "#define dK ..."). Find the dK value     
  resulting in best performance (for a given number of ranks - 32 on graham).                        
* Master (rank 0) will only distribute the workload to slaves (the rest of ranks) chunk by chunk on \
a                                                                                                    
  "first come - first served" basis and collect the results; it will not participate in prime number\
 counting itself.                                                                                    
* It is convenient to use the MPI_SOURCE element of the MPI_Status structure, in conjunction with    
  MPI_ANY_SOURCE parameter. For example, if you used MPI_Recv(...,MPI_ANY_SOURCE,....,&status),      
  then status.MPI_SOURCE will tell you the actual rank of the source.                                
* It is sufficient to only use the following MPI functions (some of them more than once):            
 - MPI_Init                                                                                          
 - MPI_Finalize                                                                                      
 - MPI_Comm_rank                                                                                     
 - MPI_Comm_size                                                                                     
 - MPI_Send                                                                                          
 - MPI_Recv                                                                                          
                                                                                                     
                                                                                                     
If you are doing Level 3 (dynamic workload balancing with the master taking part in calculations; ma\
ximum grade 100%):                                                                                   
* Going from KMAX to KMIN will facilitate dynamic workload balancing.                                
* Introduce a chunk parameter dK (can be a macro parameter: "#define dK ..."). Find the dK value     
  resulting in best performance (for a given number of ranks - 32 on graham).                        
* At this level, master rank (0) not only distributes the workload to slaves, chunk by chunk,        
  on a "first come - first served" basis, and collects the results, but also takes part in           
  counting the number of primes itself. The mode of operation for the master could be:               
  (a) check if there was a request from a slave for the next chunk, if yes - go to (c), if no - go t\
o (b)                                                                                                
  (b) process a single K prime candidate itself, then go to (a)                                      
  (c) send the next chunk to the slave, and go to (a).                                               
* It is convenient to use the MPI_SOURCE element of the MPI_Status structure, in conjunction with    
  MPI_ANY_SOURCE parameter. For example, if you used MPI_Recv(...,MPI_ANY_SOURCE,....,&status),      
  then status.MPI_SOURCE will tell you the actual rank of the source.                                
* It is sufficient to only use the following MPI functions (some of them more than once):            
 - MPI_Init                                                                                          
 - MPI_Finalize                                                                                      
 - MPI_Comm_rank                                                                                     
 - MPI_Comm_size                                                                                     
 - MPI_Send                                                                                          
 - MPI_Recv                                                                                          
 - MPI_Irecv                                                                                         
 - MPI_Test                                                                                          
                                                                                                     
                                                                                                     
                                                                                                     
Compiling instructions:                                                                              
                                                                                                     
 - Serial code:                                                                                      
  icc -O2 primes_count.c -o primes_count

  - MPI code:                                                                                         
  mpicc -O2 primes_count.c -o primes_count                                                           
   To run:                                                                                           
  mpirun -np 32 ./primes_count                                                                       
                                                                                                     
*/

/*
******************************** LEVEL 3 ********************************
*** Dynamic workload balancing with master taking part in calculation ***

Final project for CSE 745 completed by Alec Mitchell #400224703
*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h> //ADDITION: Required for MPI

#define dk 1000 //ADDITION: Chunk size for dynamic distribution

// Range of k-numbers for primes search:                                                             
#define KMIN 1
// Should be smaller than 357,913,941 (because we are using signed int)                              
#define KMAX 10000000


/* Subtract the `struct timeval' values X and Y,                                                     
   storing the result in RESULT.                                                                     
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!                                                                              

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.                                                             
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                   


int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, success;
  int xmax, ymax, x, y, k, j, count;
   
  //ADDITION: Variable for local process rank my_rank and total number of processes p
  int my_rank, p;

  MPI_Init(&argc,&argv); //ADDITION: Initialize MPI section
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //ADDITION: Function to obtain local rank
  MPI_Comm_size(MPI_COMM_WORLD, &p); //ADDITION: Function to obtain total processes

  if (my_rank == 0) { /*
  ADDITION: Allows only the master process to call gettimeofday(...)
  and perform the role of scheduler.  As per Level 3, master process also contributes to
  prime counting while idling.
  */
    gettimeofday (&tdr0, NULL);

    /*ADDITION: Running tracker to know what k values have already been checked.
      Begins at a value dk*(p-1) less than KMAX due to predetermined initial ranges.*/
    int k_run = KMAX-dk*(p-1);

    int temp_count; //ADDITION: Count for a single range received from any rank
    int recv_flag; //ADDITION: Flag to decide if receive was completed
    MPI_Request req; //ADDITION: Non-blocking receive request
    MPI_Status status; //ADDITION: Status of non-blocking receive
    count = 0; //ADDITION: Initialize total count tracked by master

    /*ADDITION: Initial non-blocking receive to accept the prime count of a single range 
      from any of the worker ranks*/
    MPI_Irecv(&temp_count, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &req);
    while (k_run >= KMIN) { //ADDITION: Loop until running tracker indicates all ks counted
      MPI_Test(&req, &recv_flag, &status); //ADDITION: Test if non-blocking receive done
      //printf("Mid Test: %d from: %d\n", recv_flag, status.MPI_SOURCE); //Debugging print
      if (recv_flag == 1) { //ADDITION: When non-blocking receive is complete
        count = count + temp_count;//ADDITION: Add count for previous range to total
        //printf("Running count:%d from: %d\n", count, status.MPI_SOURCE); //Debugging print
        int k_local[2] = {k_run-dk+1, k_run}; //Determine new range using dk and running k

        //ADDITION: If either of the new k range bounds extend passed minimum k
        if (k_local[0] < KMIN) {k_local[0] = KMIN;} 
        if (k_local[1] < KMIN) {k_local[1] = KMIN;}
        k_run = k_run-dk; //ADDITION: Update running k tracker

        //ADDITION: Send working with rank from status the new k range it will process
        MPI_Send(&k_local,2,MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        if (k_run > KMIN) { //ADDITION: When there are still more ks to distribute
          //ADDITION: Post another non-blocking receive to accept prime count of certain range
          MPI_Irecv(&temp_count, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &req);
        }
        recv_flag = 0; //ADDITION: Reset flag for receiving (not necessary, just safeguard)
      } else { //ADDITION: When non-blocking receive is not yet complete
        k = k_run; //ADDITION: Set k for master rank to count primes of and contribute

        //COPIED: Same as count algorithm for worker ranks
        // testing "-1" and "+1" cases:
        for (j=-1; j<2; j=j+2) 
          {
            // Prime candidate:                                                                        
            x = 6*k + j;
            // We should be dividing by numbers up to sqrt(x):                                         
            ymax = (int)ceil(sqrt((double)x));

            // Primality test:                                                                         
            for (y=3; y<=ymax; y=y+2)
              {
                // Tpo be a success, the modulus should not be equal to zero:                          
                success = x % y;
                if (!success)
                  break;
              }

            if (success)
              {
                count++;
              }
          }

        k_run--; //ADDITION: Update running k tracker 
      }
    }
    int k_local[2] = {KMIN, KMIN}; //ADDITION: Set final range to be sent to workers
    for (int i=0; i<p-1; i++){ //ADDITION: Iterate times equal to the number of workers
      //ADDITION: Receive last count from the workers now that all ks distributed
      MPI_Recv(&temp_count, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
      //printf("Final Receive: %d\n", status.MPI_SOURCE); //Debugging print

      //ADDITION: Send final range to workers so they know the work is complete
      MPI_Send(&k_local,2,MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      count = count + temp_count; //ADDITION: Add to total the count from last range
      //printf("count:%d\n", count); //Debugging print
    }
  } else { //ADDITION: For ranks that are workers and not the scheduler
    //ADDITION: Initial range to complete before receiving additional tasks
    int k_local[2] = {KMAX-dk*(my_rank)+1,KMAX-dk*(my_rank-1)}; 
    while (k_local[1] != KMIN) { //ADDITION: Loop until upper bound of range received is KMIN
      //printf("Range: %d to %d from: %d\n", k_local[0], k_local[1], my_rank); //Debugging print
      count = 0; //ADDITION: Initialize local prime count for specific k range
      for (k=k_local[0]; k<=k_local[1]; k++) { //MODIFIED: Iterate over local range of ks

        // testing "-1" and "+1" cases:                                                                
        for (j=-1; j<2; j=j+2)
          {
            // Prime candidate:                                                                        
            x = 6*k + j;
            // We should be dividing by numbers up to sqrt(x):                                         
            ymax = (int)ceil(sqrt((double)x));

            // Primality test:                                                                         
            for (y=3; y<=ymax; y=y+2)
              {
                // Tpo be a success, the modulus should not be equal to zero:                          
                success = x % y;
                if (!success)
                  break;
              }

            if (success)
              {
                count++;
              }
          }

      
      }
      MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); //ADDITION: Send local count to master
      //Debugging print
      //printf("Sending: %d in %d to %d from: %d\n", count, k_local[0], k_local[1], my_rank);
      //ADDITION: Receive the next range to work on
      MPI_Recv(&k_local,2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    //printf("DONE! %d\n", my_rank ); //Debugging print
  }
  if (my_rank == 0) { 
  //ADDITION: Directs master process to complete timing and print prime count
    gettimeofday (&tdr1, NULL);
    tdr = tdr0;
    timeval_subtract (&restime, &tdr1, &tdr);
    printf ("N_primes: %d\n", count);
    printf ("time: %e\n", restime);
  }
  MPI_Finalize(); //ADDITION: Close/end MPI section
  //--------------------------------------------------------------------------------  


  return 0;

}

