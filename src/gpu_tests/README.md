# Performing Calculations on a GPU

This example was taken from https://developer.apple.com/documentation/metal/basic_tasks_and_concepts/performing_calculations_on_a_gpu

## Overview

This code snipped uses Apples' Metal Shading Language (MSL) to run a convolution operation on the GPU. The performance of the M1 GPU is compared with a NVidia Tesla M60 from Skoltechs' Zhores supercomputer, as well as GPUs from Google Colab.

## Code

The M1 code was implemented in MSL, the auxiliary code is implemented in Objective-C. A CUDA implementation of the convolution operation was also included.

## Experiment setup

For the tests, the *convolution filter* from the HPC assignment 4 was ported to MSL. Then, this convolution filter operation will be run on images with three sizes:
*  8k (7680 x 4320)
* 16k (15360 x 8640)
* 32k (30720 x 17280)

The main performance metric is the time it takes for the machine to complete the computation.

### Tested devices

For the test, three devices are compared:
* Apple M1 SoC computer
* Nvidia Tesla M60 CUDA GPU (Zhores Sandbox)
* Nvidia Tesla T4 CUDA GPU (Google Colab)

For the CUDA cores, two time measurements are taken: One with the memory copying operations included, and one without. Since the M1 shares all of its memory between the CPU and the GPU, such a measurement is not necessary for the M1.

## Results

| Device | 8k image | 16k image | 32k image |
|--------|----------|-----------|-----------|
| Apple M1 |27.862ms|110.968ms|693.968ms|
| Nvidia Tesla M60|797.6ms (no copy), 885.95ms (copy)|2534.57ms (no copy), 2882.694ms (copy)|13437.42ms (no copy), 15118.365ms (copy)|
| Nvidia Tesla T4|195.3ms (no copy), 284.72ms (copy)|748.17ms (no copy), 1082.1ms (copy)|4910.34ms (no copy), 6243.77ms (copy)|
