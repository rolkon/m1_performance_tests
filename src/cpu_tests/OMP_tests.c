#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void zero_init_matrix(double ** matrix, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i) {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}

int main(int argc, char ** argv)
{
    // argv[1] – size of the square matrices and vectors
    // argv[2] – num threads

    // UTILS
    unsigned long int N = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));
    double start, end;
    int i, j, k;

    // TEST 1: VECTOR SCALAR PRODUCT
    printf("\n***** Starting Test 1: Vector Scalar Product *****\n");

    // Set the size of the vector
    printf("Vector size is %d\n", N);

    // Alloc the vectors
    double * x = (double *)malloc(N * sizeof(double));
    double * y = (double *)malloc(N * sizeof(double));
    double z = 0.0;

    // Sequential
    // Init vectors
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        x[i] = rand() / RAND_MAX;
        y[i] = rand() / RAND_MAX;
    }

    start = omp_get_wtime();

    for (i = 0; i < N; i++) {
        z += x[i] * y[i];
    }

    end = omp_get_wtime();

    printf("Sequential: %f milliseconds.\n", (end - start) * 1000);

    // Parallel
    // Init vectors
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        x[i] = rand() / RAND_MAX;
        y[i] = rand() / RAND_MAX;
    }

    start = omp_get_wtime();

# pragma omp parallel for shared(x, y) reduction(+:z) private(i, j, k)
    for (i = 0; i < N; i++) {
        z += x[i] * y[i];
    }
    end = omp_get_wtime();

    printf("Parallel: %f milliseconds.\n", (end - start) * 1000);

    free(x);
    free(y);

    // TEST 2: MATRIX-BY-VECTOR MULTIPLICATION
    printf("\n***** Starting Test 2: Matrix-by-Vector Multiplication *****\n");
    
    // Print the size of the matrix
    printf("Matrix size is %d\n", N);
    
    // Alloc matrix and vectors
    double ** A;
    A = malloc_matrix(N);
    x = (double *)malloc(N * sizeof(double));
    double * b = (double *)malloc(N * sizeof(double));

    // Sequential
    // Init matrix and vectors
    rand_init_matrix(A, N);
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        x[i] = rand() / RAND_MAX;
        b[i] = 0.0;
    }

    start = omp_get_wtime();

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            b[i] += A[i][j] * x[j];
        }
    }
    end = omp_get_wtime();

    printf("Sequential: %f milliseconds.\n", (end - start) * 1000);

    // Parallel
    // Init matrix and vectors
    rand_init_matrix(A, N);
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        x[i] = rand() / RAND_MAX;
        b[i] = 0.0;
    }

    start = omp_get_wtime();

# pragma omp parallel for collapse(2) shared(A, x) private(i, j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
#pragma omp atomic
            b[i] += A[i][j] * x[j];
        }
    }
    end = omp_get_wtime();

    printf("Parallel: %f milliseconds.\n", (end - start) * 1000);

    free_matrix(A, N);
    free(x);
    free(b);

    // TEST 3: MATRIX-BY-MATRIX MULTIPLICATION
    printf("\n***** Starting Test 3: Matrix-by-Matrix Multiplication *****\n");

    // Print the size of the matrix
    printf("Matrix size is %d\n", N);
    
    // Alloc matrices
    double ** B, ** C;

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);

    // Sequential ijk
    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();

    printf("Sequential ijk: %f milliseconds.\n", (end - start) * 1000);

    // Parallel ijk
    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();
    
# pragma omp parallel for collapse(3) shared(A, B) private(i, j, k)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
#pragma omp atomic
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    end = omp_get_wtime();

    printf("Parallel ijk: %f milliseconds.\n", (end - start) * 1000);

    // Sequential jik
    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();

    printf("Sequential jik: %f milliseconds.\n", (end - start) * 1000);

    // Sequential kij
    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

    for (k = 0; k < N; k++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();

    printf("Sequential kij: %f milliseconds.\n", (end - start) * 1000);

    // Parallel jik
    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

# pragma omp parallel for collapse(3) shared(A, B) private(i, j, k)
    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
            for (k = 0; k < N; k++) {
#pragma omp atomic
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();

    printf("Parallel jik: %f milliseconds.\n", (end - start) * 1000);

    // Parallel kij
    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    start = omp_get_wtime();

# pragma omp parallel for collapse(3) shared(A, B) private(i, j, k)
    for (k = 0; k < N; k++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
#pragma omp atomic
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();

    printf("Parallel kij: %f milliseconds.\n", (end - start) * 1000);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}