/*
See LICENSE folder for this sample’s licensing information.

Abstract:
An app that performs a simple calculation on a GPU.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalFilter.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalFilter* filter = [[MetalFilter alloc] initWithDevice:device];
        
        // Load image file into the buffers
        [filter loadImageFromPathIntoBuffer:@"Data/32k_death_valley.ppm"];
        
        // init the kernel
        [filter initializeSmoothingKernel5x5];
        
        // Send a command to the GPU to perform the calculation.
        [filter sendComputeCommand];
        
        [filter writeImageFromBufferToPath:@"Data/32k_death_valley_blurry.ppm"];
        
//        [filter printDebugBufferFrom:0 to:81];

        NSLog(@"Execution finished");
    }
    return 0;
}
