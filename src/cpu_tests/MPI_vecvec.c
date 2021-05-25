// The code is adapted from https://github.com/imsure/parallel-programming/blob/master/matrix-multiplication/mpi-mm.c

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int TAG = 777;

int main(int argc, char ** argv) {
  double *x, *c;
  double res;
  double elapsed_time;
  int numElements, offset, stripSize, rank, size, N, i, j, k;
  
  // Set up MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Size of the matrix
  N = atoi(argv[1]);

  if (rank == 0) {
    printf("\n***** Starting Test 1: Vector Scalar Product *****\n");
    printf("Vector size is %d, number of processes is %d.\n", N, size);
  }

  // Allocate x and c on main and worker processes
  if (rank == 0) {
    x = (double *) malloc (sizeof(double) * N);
    c = (double *) malloc (sizeof(double) * N);
  }
  else {
    x = (double *) malloc (sizeof(double) * N / size);
    c = (double *) malloc (sizeof(double) * N / size);
  }
  

  stripSize = N / size;

  // Init x and c with random numbers for each process
  if (rank == 0) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
      x[i] = rand() / RAND_MAX;
      c[i] = rand() / RAND_MAX;
    }
  } else {
    srand(time(NULL));
    for (int i = 0; i < stripSize; i++) {
      x[i] = rand() / RAND_MAX;
      c[i] = rand() / RAND_MAX;
    }
  }

  if (rank == 0) {
    elapsed_time = MPI_Wtime();
  }

  if (rank == 0) {
    // Send parts of x and c to workers
    offset = stripSize;
    numElements = stripSize;
    for (i=1; i<size; i++) {
      MPI_Send(&x[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
      MPI_Send(&c[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
      offset += stripSize;
    }
  }
  else {
    // Receive parts of x and c
    MPI_Recv(x, stripSize, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(c, stripSize, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Compute scalar product
  for (i = 0; i < stripSize; i++) {
      res += x[i] * c[i];
  }

  // Main process receives results
  if (rank == 0) {
    for (i = 1; i < size; i++) {
      double temp;
      MPI_Recv(&temp, 1, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      res += temp;
    }
  }
  else {
    // Send results to the main process
    MPI_Send(&res, 1, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    elapsed_time = MPI_Wtime() - elapsed_time;
    printf("Parallel: %f milliseconds.\n", elapsed_time * 1000);
  }
  
  MPI_Finalize();
  return 0;
}

