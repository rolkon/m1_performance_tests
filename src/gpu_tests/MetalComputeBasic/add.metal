/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A shader that adds two arrays of floats.
*/

#include <metal_stdlib>

//struct defined here as well - not ideal, but I don't see another way
typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

using namespace metal;
/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.
kernel void add_arrays(device const PPMPixel* inA,
                       device const PPMPixel* inB,
                       device PPMPixel* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index].red = inA[index].red + inB[index].red;
    result[index].green = inA[index].green + inB[index].green;
    result[index].blue = inA[index].blue + inB[index].blue;
    //result[index] = inA[index] + inB[index];
}

kernel void switch_color_channels(device const PPMPixel* in,
                        device PPMPixel* out,
                        uint index [[thread_position_in_grid]])
{
    out[index].red = in[index].green;
    out[index].green = in[index].blue;
    out[index].blue = in[index].red;
}
