## M1 CPU Tests with OpenMP and MPI
---

### Installation & Environment Setup

**Note:** for better clarity instructions below contain `arch -arm64` or `arch -x86_64` before every Terminal command to indicate whether the command is executed on native ARM or with Rosetta. If you understand what architecture your Terminal works on at start (you can check at Applications -> Terminal -> Get Info -> Open using Rosetta), you can specify the `arch` keyword only when you wish to work with another architecture.

1. Install the ARM version of `brew` with the script they provide on the [official website](https://brew.sh).
    ```bash
    arch -arm64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    **Note:** by default the directory with ARM `brew` tools is `/opt/homebrew/bin/brew`.
    
    **Note:** as of May 2021 about 70% of all packages are transported to the ARM version of `brew`. Luckily, OpenMP and MPI libraries are among them.
    
    **Help:** [this thread](https://github.com/Homebrew/discussions/discussions/149) can be useful for dealing with potential issues. 
    
2. Install Intel version of `brew` with Rosetta. 
    ```bash
    arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    
    **Note:** by default the directory with Intel `brew` tools is `/usr/local/bin/`.
    
3. Install OpenMP and GCC with `brew`.
    - ARM
        ```bash
        arch -arm64 brew install libomp
        arch -arm64 brew install gcc
        ```
    - Intel with Rosetta
        ```bash
        arch -x86_64 brew install libomp
        arch -x86_64 brew install gcc
        ```
    **Note:** as of May 2021 the version of GCC installed is `gcc-11`.
4. Install MPI with `brew`.
    - ARM
        ```bash
        arch -arm64 brew install open-mpi
        ```
    - Intel with Rosetta
        ```bash
        arch -x86_64 brew install open-mpi
        ```

### Compilation

Below are the examples of OpenMP and MPI files compilation on both native ARM and Rosetta. If `brew` is installed in non-default directory, change the corresponding path to `gcc-11`, `mpicc` and `mpirun`.

- OpenMP compilation.
    - ARM
        ```bash
        arch -arm64 /opt/homebrew/bin/gcc-11 -fopenmp omp.cpp
        arch -arm64 ./a.out
        ```
    - Intel with Rosetta
        ```bash
        arch -x86_64 /usr/local/bin/gcc-11 -fopenmp omp.cpp
        arch -x86_64 ./a.out
        ```
- MPI compilation.
    - ARM
        ```bash
        arch -arm64 mpicc mpi.cpp
        arch -arm64  mpirun -n 4 ./a.out
        ```
    - Intel with Rosetta
        ```bash
        arch -x86_64 /usr/local/bin/mpicc mpi.cpp 
        arch -x86_64 /usr/local/bin/mpirun -n 4 ./a.out
        ```
    
### Experiments Description

The experiments test the clock time of one [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) operation for each level: 
- Level 1: Vector scalar product.
- Level 2: Matrix-by-vector product.
- Level 3: Matrix-by-matrix product.

The routines for conducting experiments are contained in the following files. The routines for MPI experiments are separated into different files to avoid issues with messy memory handling.

**Note:** for MPI experiments the matrix size **must** be divisible by the number of processes. 

- `OMP_tests.c`
    Provides the clock time for sequential and OMP-parallel versions for all three BLAS levels. Level 3 is done for 3 different indexing ways: ijk, jik, kij.

    ```bash
    arch -arm64 ./a.out N k
    ```
    where `N` is the size of matrices and vectors, `k` is the number of threads.
- `MPI_vecvec.c`
    Provides the clock time for MPI-parallel version for BLAS-1.
- `MPI_matvec.c`
    Provides the clock time for MPI-parallel version for BLAS-2.
- `MPI_matmul.c`
    Provides the clock time for MPI-parallel version for BLAS-3.

    ```bash
    arch -x86_64 /usr/local/bin/mpirun -np 4 ./a.out N
    ```
    where `N` is the size of matrices. 

### Results

For OpenMP the number of threads was set equal to the number of cores. For MPI the number of processes was set equal to the number of cores. `N` in each was chosen to be large enough and divisible by the number of cores. The tuples are `(BLAS1, BLAS2, BLAS3)` time in milliseconds. The `BLAS3` benchmark is for ijk indexing. 

**Notes:**
1. For some reason sometimes ARM and Rosetta M1 throws a segmentation error after executing the MPI scripts. This error does not affect the script's performance. 
2. In Colab you have to test with `!mpirun --allow-run-as-root -np 2 ./a.out 512`.


| Computer | OS | Architecture |  N | Sequential | OpenMP | MPI |
| -------- | --- | ------------ | --- | ---------- | --- | --- |
| MacBook Pro, M1 (2020) | macOS 11.3.1 | ARM @ 3.20GHz, 8 cores (2020) | 512 | (0.005, 2.256, 508.996) | (0.294, 0.837, 223.965) | (10.724, 8.193, 116.172) |
| MacBook Pro, M1 (2020) | macOS 11.3.1 | Rosetta @ 3.20GHz, 8 cores (2020) | 512 | (0.004, 1.110, 493.877) | (0.381, 0.529, 180.055) | (0.84, 5.723, 123.383) |
| Colab | Ubuntu 18.04.5 LTS | Intel(R) Xeon(R) CPU @ 2.20GHz, 2 cores | 512 | (0.0059, 1.657, 1627.569) | (0.111, 2.524, 1717.255) | (0.062, 1.922, 1545.168) |
| HP Laptop 15-db0460ur (2019) | Ubuntu 20.04.2 LTS | AMD A9-9425 RADEON R5, 5 COMPUTE CORES 2C+3G @ 3.1GHz, 2 cores (2016) | 512 | (0.007, 1.417, 3028.264) | (0.133, 3.417, 5844.955) | (0.029, 2.783, 3668.932) |





    
    
    
    
    
    