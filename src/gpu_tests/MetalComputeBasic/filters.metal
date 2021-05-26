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

typedef struct {
    unsigned int x, y;
    unsigned int size;
    unsigned int memory;
} PPMImageShape;

using namespace metal;

//kernel void switch_color_channels(device const PPMPixel* in,
//                                  device PPMPixel* out,
//                                  device uint* debug,
//                                  uint index [[thread_position_in_grid]])
//{
//    out[index].red = in[index].green;
//    out[index].green = in[index].blue;
//    out[index].blue = in[index].red;
//
//    debug[index] = index;
//}

//convolution operation for a 5x5 kernel
kernel void convolution(device const PPMPixel* img_in,
                        device PPMPixel* img_out,
                        device float* kern,
                        device PPMImageShape* img_shape,
                        device uint* debug,
                        uint index [[thread_position_in_grid]])
{
    PPMPixel conv_buffer[9*9];
    
    unsigned int coord_x = index % img_shape->x;
    unsigned int coord_y = index / img_shape->x;
    
    //debug[index] = coord_x;

    int k_size = 9;
    int k_offset = 4;
    
    for(int row=-k_offset; row<=k_offset; row++)
    {
        for(int col=-k_offset; col<=k_offset; col++)
        {
            if(coord_x+col < 0 || coord_x+col >= img_shape->x || coord_y+row < 0 || coord_y+row >= img_shape->y)
            {
                conv_buffer[(row+k_offset)*k_size+(col+k_offset)].red    = 0;
                conv_buffer[(row+k_offset)*k_size+(col+k_offset)].green  = 0;
                conv_buffer[(row+k_offset)*k_size+(col+k_offset)].blue   = 0;
            }else{
                conv_buffer[(row+k_offset)*k_size+(col+k_offset)].red    = img_in[(coord_y+row)*img_shape->x+(coord_x+col)].red;
                conv_buffer[(row+k_offset)*k_size+(col+k_offset)].green  = img_in[(coord_y+row)*img_shape->x+(coord_x+col)].green;
                conv_buffer[(row+k_offset)*k_size+(col+k_offset)].blue   = img_in[(coord_y+row)*img_shape->x+(coord_x+col)].blue;
            }
        }
    }
    
    if(index == 0)
    {
        for(int i=0; i<81; i++)
        {
            debug[i] = conv_buffer[i].blue;
        }
    }

    float sum_red     = 0.0;
    float sum_green   = 0.0;
    float sum_blue    = 0.0;

    for(int i=0; i<9*9; i++)
    {
        sum_red     += (float)conv_buffer[i].red    * kern[i];
        sum_green   += (float)conv_buffer[i].green  * kern[i];
        sum_blue    += (float)conv_buffer[i].blue   * kern[i];
    }

    img_out[index].red      = (char)sum_red;
    img_out[index].green    = (char)sum_green;
    img_out[index].blue     = (char)sum_blue;
}
