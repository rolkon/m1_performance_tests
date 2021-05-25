// The code is adapted from https://github.com/imsure/parallel-programming/blob/master/matrix-multiplication/mpi-mm.c

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int TAG = 777;

int main(int argc, char ** argv) {
  double **A, **B, **C, *tmp;
  double elapsed_time;
  int numElements, offset, stripSize, rank, size, N, i, j, k;
  
  // Set up MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Size of the matrix
  N = atoi(argv[1]);

  if (rank == 0) {
    printf("\n***** Starting Test 3: Matrix-by-Matrix Multiplication *****\n");
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
  
  // Allocate B on all processes
  tmp = (double *) malloc (sizeof(double ) * N * N);
  B = (double **) malloc (sizeof(double *) * N);
  for (i = 0; i < N; i++) B[i] = &tmp[i * N];
  
  // Allocate C on main and worker processes
  if (rank == 0) {
    tmp = (double *) malloc (sizeof(double ) * N * N);
    C = (double **) malloc (sizeof(double *) * N);
    for (i = 0; i < N; i++) C[i] = &tmp[i * N];
  }
  else {
    tmp = (double *) malloc (sizeof(double ) * N * N / size);
    C = (double **) malloc (sizeof(double *) * N / size);
    for (i = 0; i < N / size; i++) C[i] = &tmp[i * N];
  }

  // Init A and B with random numbers
  if (rank == 0) {
    srand(time(NULL));
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        A[i][j] = rand() / RAND_MAX;
        B[i][j] = rand() / RAND_MAX;
      }
    }
  }

  stripSize = N / size;

  // Init C with zeros for each process
  for (i = 0; i < stripSize; i++) {
    for (j = 0; j < N; j++) {
      C[i][j] = 0.0;
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
  MPI_Bcast(B[0], N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Matrix multiplication
  for (i = 0; i < stripSize; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < N; k++) {
	        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  // Main process receives results
  if (rank == 0) {
    offset = stripSize; 
    numElements = stripSize * N;
    for (i=1; i<size; i++) {
      MPI_Recv(C[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      offset += stripSize;
    }
  }
  else {
    // Send results to the main process
    MPI_Send(C[0], stripSize * N, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    elapsed_time = MPI_Wtime() - elapsed_time;
    printf("Parallel ijk: %f milliseconds.\n", elapsed_time * 1000);
  }
  
  MPI_Finalize();
  return 0;
}

