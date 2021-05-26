# Performing Calculations on a GPU

This example was taken from https://developer.apple.com/documentation/metal/basic_tasks_and_concepts/performing_calculations_on_a_gpu

## Overview

This code snipped uses Apples' Metal Shading Language (MSL) to run a convolution operation on the GPU. The performance of the M1 GPU is compared with a NVidia Tesla M60 from Skoltechs' Zhores supercomputer, as well as GPUs from Google Colab.

## Code

The shader is implemented in MSL, the auxiliary code is implemented in Objective-C.

## Experiment setup

For the tests, the *median filter* from the HPC assignment 4 will be ported to MSL. Then, this median filter operation will be run on images with various sizes.

The implemented median filter has a complexity of O(n x m), with n being the size of the kernel, and m being the size of the sub-grid of the image for each GPU thread.

Two performance metrics will be analyzed:
1. Time it takes for the calculation
2. Energy consumption during the calculation

## Results
