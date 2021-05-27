# M1 performance tests
Repository for conducting performance tests on Apple's new M1 chip

## Tests to be conducted:
* Comparision between native silicon, as well as Rosetta x86 abstraction with comparative devices

### Intel / AMD CPU vs M1 CPU

1. x86 CPU vs M1 CPU (ARM) Single Core
2. x86 CPU vs M1 CPU (ARM) OMP
3. x86 CPU vs M1 CPU (ARM) MPI
4. x86 CPU vs M1 CPU (Rosetta) Single Core
5. x86 CPU vs M1 CPU (Rosetta) OMP
6. x86 CPU vs M1 CPU (Rosetta) MPI

### CUDA vs M1 GPU

1. CUDA vs M1 GPU (without copying buffer, just computation)
2. CUDA vs M1 GPU (with copying buffer - check speedup as compared to "slower" transfer in CUDA)
