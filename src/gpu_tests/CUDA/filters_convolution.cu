#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ppm_file_reader.h"
#include "kernels.h"

//base of this code is taken from Roland Konlechner HPC 2021 assignment 4: CUDA convolution kernel
//make one convolution engine which accepts arbitrary 9x9 kernels
__global__ void Convolution(PPMPixel* img_in, PPMPixel* img_out, float* kernel, int size_x, int size_y)
{
    int globalIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int nr_blocks = blockDim.x * blockDim.y * blockDim.z;

    globalIdx += nr_blocks*(blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x);

    //divide rows of image amongst blocks, and colums amongst threads within blocks
    float heightPerBlock = (float)size_y / (float)gridDim.x;
    float widthPerThread = (float)size_x / (float)blockDim.x;

    int blockStart = round(blockIdx.x * heightPerBlock);
    int blockStop  = round((blockIdx.x+1) * heightPerBlock - 1);
    int threadStart = round(threadIdx.x * widthPerThread);
    int threadStop = round((threadIdx.x+1) * widthPerThread - 1);

    //allocate local kernel buffer and copy part of image. For the edge cases, fill the buffer with black
    PPMPixel conv_buffer[81];
    float sum_red;
    float sum_green;
    float sum_blue;

    int k_offset = 4;
    int k_size = 9;

    for(int row=blockStart; row<=blockStop; row++)
    {
        for(int col=threadStart; col<=threadStop; col++)
        {
            for(int kern_row=-k_offset; kern_row<=k_offset; kern_row++)
            {
                for(int kern_col=-k_offset; kern_col<=k_offset; kern_col++)
                {
                    if(row+kern_row < 0 || row+kern_row > size_y || col+kern_col < 0 || col+kern_col > size_x)
                    {                        
                        conv_buffer[(kern_row+k_offset)*k_size+(kern_col+k_offset)].red   = 0;
                        conv_buffer[(kern_row+k_offset)*k_size+(kern_col+k_offset)].green = 0;
                        conv_buffer[(kern_row+k_offset)*k_size+(kern_col+k_offset)].blue  = 0;
                    }else{
                        conv_buffer[(kern_row+k_offset)*k_size+(kern_col+k_offset)].red   = img_in[(row+kern_row)*size_x+(col+kern_col)].red;
                        conv_buffer[(kern_row+k_offset)*k_size+(kern_col+k_offset)].green = img_in[(row+kern_row)*size_x+(col+kern_col)].green;
                        conv_buffer[(kern_row+k_offset)*k_size+(kern_col+k_offset)].blue  = img_in[(row+kern_row)*size_x+(col+kern_col)].blue;
                    }
                }
            }
            sum_red     = 0.0;
            sum_green   = 0.0;
            sum_blue    = 0.0;
            
            //do the convolution on the local buffer and write result to output
            for(int i=0; i<81; i++)
            {
                sum_red    += (float)conv_buffer[i].red   * kernel[i];
                sum_green  += (float)conv_buffer[i].green * kernel[i];
                sum_blue   += (float)conv_buffer[i].blue  * kernel[i];
            }
            img_out[row*size_x+col].red     = (char)sum_red;
            img_out[row*size_x+col].green   = (char)sum_green;
            img_out[row*size_x+col].blue    = (char)sum_blue;
        }
    }
}


int main(){
    PPMImage* h_img[3];
    h_img[0] = readPPM("../data/8k_mountains.ppm");
    h_img[1] = readPPM("../data/16k_forest.ppm");
    h_img[2] = readPPM("../data/32k_death_valley.ppm");

    char* outputNames[3] = {"../data/8k_mountains_blurry_cuda.ppm", "../data/16k_forest_blurry_cuda.ppm", "../data/32k_death_valley_blurry_cuda.ppm"};

    // from https://stackoverflow.com/questions/28112485/how-to-select-a-gpu-with-cuda/28113186

    cudaSetDevice(1);

    cudaEvent_t start_copy, stop_copy, start_no_copy, stop_no_copy;

    cudaEventCreate(&start_copy);
    cudaEventCreate(&start_no_copy);
    cudaEventCreate(&stop_copy);
    cudaEventCreate(&stop_no_copy);

    for(int i=0; i<3; i++)
    {
        printf("Test run %d\n", i);

        long int img_size = h_img[i]->x * h_img[i]->y * sizeof(PPMPixel);
        int kernel_size = 9*9*sizeof(float);

        PPMPixel* d_img_data;
        PPMPixel* d_res_data;
        PPMPixel* h_res_data;

        float* h_kernel_box_blur = kernel_box_blur_9x9();
        float* d_kernel_box_blur;

        h_res_data = (PPMPixel*)malloc(img_size);
        cudaMalloc(&d_img_data, img_size);
        cudaMalloc(&d_res_data, img_size);
        cudaMalloc(&d_kernel_box_blur, kernel_size);

        cudaEventRecord(start_copy);

        cudaMemcpy(d_img_data, h_img[i]->data, img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel_box_blur, h_kernel_box_blur, kernel_size, cudaMemcpyHostToDevice);

        //data on host is not needed anymore, let host image reference the result data
        free(h_img[i]->data);
        h_img[i]->data = h_res_data;

        cudaEventRecord(start_no_copy);

        Convolution<<<32, 1024>>>(d_img_data, d_res_data, d_kernel_box_blur, h_img[i]->x, h_img[i]->y);
        
        cudaEventRecord(stop_no_copy);
        
        cudaMemcpy(h_res_data, d_res_data, img_size, cudaMemcpyDeviceToHost);
        
        cudaEventRecord(stop_copy);
        
        cudaEventSynchronize(stop_no_copy);
        cudaEventSynchronize(stop_copy);
        cudaDeviceSynchronize();

        float milliseconds_copy = 0;
        float milliseconds_no_copy = 0;

        cudaEventElapsedTime(&milliseconds_copy, start_copy, stop_copy);
        cudaEventElapsedTime(&milliseconds_no_copy, start_no_copy, stop_no_copy);

        writePPM(outputNames[i], h_img[i]);

        printf("\tTime elapsed (no copy) [ms]:\t%f\n", milliseconds_no_copy);
        printf("\tTime elapsed (copy) [ms]:\t%f\n", milliseconds_copy);

        free(h_kernel_box_blur);
        free(h_res_data);
        cudaFree(d_img_data);
        cudaFree(d_res_data);
        cudaFree(h_kernel_box_blur);
    }

    free(h_img[0]);
    free(h_img[1]);
    free(h_img[2]);

