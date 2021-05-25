// The code is adapted from https://github.com/imsure/parallel-programming/blob/master/matrix-multiplication/mpi-mm.c

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int TAG = 777;

int main(int argc, char ** argv) {
  double **A, *x, *c, *tmp;
  double elapsed_time;
  int numElements, offset, stripSize, rank, size, N, i, j, k;
  
  // Set up MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Size of the matrix
  N = atoi(argv[1]);

  if (rank == 0) {
    printf("\n***** Starting Test 2: Matrix-by-Vector Multiplication *****\n");
    printf("Matrix size is %d, number of processes is %d.\n", N, size);
  }
  
  // Allocate A on main and worker processes
  if (rank == 0) {
    tmp = (double *) malloc (sizeof(double ) * N * N);
    A = (double **) malloc (sizeof(double *) * N);
    for (i = 0; i < N; i++) A[i] = &tmp[i * N];
  }
  else {
    tmp = (double *) malloc (sizeof(double ) * N * N / size);
    A = (double **) malloc (sizeof(double *) * N / size);
    for (i = 0; i < N / size; i++) A[i] = &tmp[i * N];
  }
  
  // Allocate x on all processes
  x = (double *) malloc (sizeof(double) * N);
  
  // Allocate c on main and worker processes
  if (rank == 0) {
    c = (double *) malloc (sizeof(double) * N);
  }
  else {
    c = (double *) malloc (sizeof(double) * N / size);
  }

  // Init A and x with random numbers
  if (rank == 0) {
    srand(time(NULL));
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        A[i][j] = rand() / RAND_MAX;
      }
      x[i] = rand() / RAND_MAX;
    }
  }

  stripSize = N / size;

  // Init c with zeros for each process
  if (rank == 0) {
    for (int i = 0; i < N; i++) {
      c[i] = 0.0;
    }
  } else {
    for (int i = 0; i < stripSize; i++) {
      c[i] = 0.0;
    }
  }

  if (rank == 0) {
    elapsed_time = MPI_Wtime();
  }

  if (rank == 0) {
    // Send parts of A to workers
    offset = stripSize;
    numElements = stripSize * N;
    for (i=1; i<size; i++) {
      MPI_Send(A[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
      offset += stripSize;
    }
  }
  else {
    // Receive parts of A
    MPI_Recv(A[0], stripSize * N, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  // Send B to every process
  MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Matrix-by-vector multiplication
  for (i = 0; i < stripSize; i++) {
    for (j = 0; j < N; j++) {
      c[i] += A[i][j] * x[j];
    }
  }

  // Main process receives results
  if (rank == 0) {
    offset = stripSize; 
    numElements = stripSize;
    for (i = 1; i < size; i++) {
      MPI_Recv(&c[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      offset += stripSize;
    }
  }
  else {
    // Send results to the main process
    MPI_Send(&c[0], stripSize, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    elapsed_time = MPI_Wtime() - elapsed_time;
    printf("Parallel: %f milliseconds.\n", elapsed_time * 1000);
  }
  
  MPI_Finalize();
  return 0;
}

