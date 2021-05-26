/*
See LICENSE folder for this sample’s licensing information.

Abstract:
An app that performs a simple calculation on a GPU.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalFilter.h"
#import <time.h>
#import <stdio.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        clock_t start, stop;
        double time_elapsed;
        
        NSArray *inputNames = @[@"../data/8k_mountains.ppm",
                                @"../data/16k_forest.ppm",
                                @"../data/32k_death_valley.ppm"];
        
        NSArray *outputNames = @[@"../data/8k_mountains_blurry_m1.ppm",
                                 @"../data/16k_forest_blurry_m1.ppm",
                                 @"../data/32k_death_valley_blurry_m1.ppm"];
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalFilter* filter = [[MetalFilter alloc] initWithDevice:device];
        
        for(int i=0; i<3; i++)
        {
            printf("Test run %d\n", i);
            
            // Load image file into the buffers
            [filter loadImageFromPathIntoBuffer:inputNames[i]];
            
            // init the kernel
            [filter initializeSmoothingKernel9x9];
            
            start = clock();
            
            // Send a command to the GPU to perform the calculation.
            [filter sendComputeCommand];
            
            stop = clock();
            
            time_elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
            
            printf("\tTime elapsed[ms]: %lf\n", time_elapsed*1000);
            
            [filter writeImageFromBufferToPath:outputNames[i]];
        }
        
//        [filter printDebugBufferFrom:0 to:81];

        NSLog(@"Execution finished");
    }
    return 0;
}
