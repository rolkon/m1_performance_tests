# M1 performance tests
Repository for conducting performance tests on Apple's new M1 chip

## Tests to be conducted:
* Comparision between native silicon and Rosetta x86 abstraction

### Series 1 (External Comparison)

#### Intel / AMD CPU vs M1 CPU

1. x86 CPU vs M1 CPU (ARM) Single Core
2. x86 CPU vs M1 CPU (ARM) OMP
3. x86 CPU vs M1 CPU (ARM) MPI
4. x86 CPU vs M1 CPU (Rosetta) Single Core
5. x86 CPU vs M1 CPU (Rosetta) OMP
6. x86 CPU vs M1 CPU (Rosetta) MPI

#### CUDA vs M1 GPU

1. CUDA vs M1 GPU (without copying buffer, just computation)
2. CUDA vs M1 GPU (with copying buffer - check speedup as compared to "slower" transfer in CUDA)

### Series 2 (Internal Comparison)

#### M1 CPU vs M1 GPU

1. M1 CPU (ARM) All cores, OMP vs M1 GPU (ARM)
2. M1 CPU (Rosetta) All cores, OMP vs M1 GPU (Rosetta)
3. M1 CPU (ARM) All cores, OMP vs M1 CPU (Rosetta) All cores, OMP
4. M1 GPU (ARM) vs M1 GPU (Rosetta)

### Testables

For each category above test the following.

1. **Baseline**: matrix multiplication of square matrices of size 10, 100, 1000.
2. ...
3. ...

### Metrics

For each category and testable above measure the following.

1. Clock time. 
2. ...
3. ...
