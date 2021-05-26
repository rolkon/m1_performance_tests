/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalFilter.h"
#import "ppm_file_reader.h"
#import <stdlib.h>

@implementation MetalFilter
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mAddFunctionPSO;

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;

    // Buffers to hold image data, as well as result after convolution.
    id<MTLBuffer> _mBufferImg;
    id<MTLBuffer> _mBufferResult;
    id<MTLBuffer> _mBufferKernel;
    id<MTLBuffer> _mBufferShape;
    id<MTLBuffer> _mBufferDebug;
    
    PPMImageShape img_shape;
}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self)
    {
        _mDevice = device;

        NSError* error = nil;

        // Load the shader files with a .metal file extension in the project

        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        if (defaultLibrary == nil)
        {
            NSLog(@"Failed to find the default library.");
            return nil;
        }

        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"convolution"];
        if (addFunction == nil)
        {
            NSLog(@"Failed to find function.");
            return nil;
        }

        // Create a compute pipeline state object.
        _mAddFunctionPSO = [_mDevice newComputePipelineStateWithFunction: addFunction error:&error];
        if (_mAddFunctionPSO == nil)
        {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return nil;
        }

        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }

    return self;
}

- (void) loadImageFromPathIntoBuffer:(NSString *)path
{
    //read image file with reader
    //convert NSString to c-string
    const char* img_path = [path UTF8String];
    PPMImage* img = readPPM(img_path);
    img_shape.x = img->x;
    img_shape.y = img->y;
    img_shape.size = img->x * img->y;
    img_shape.memory = img_shape.size * sizeof(PPMPixel);
    
    //initialize buffer by without allocation by referencing the existing pixel data
    _mBufferImg = [_mDevice newBufferWithBytesNoCopy:(void*)img->data length:img_shape.memory options:MTLResourceStorageModeShared deallocator:nil];
    
    //allocate result, kernel and size buffers
    _mBufferResult  = [_mDevice newBufferWithLength:img_shape.memory options:MTLResourceStorageModeShared];
    _mBufferShape   = [_mDevice newBufferWithLength:sizeof(PPMImageShape) options:MTLResourceStorageModeShared];
    
    //init buffer for debug purposes
    _mBufferDebug   = [_mDevice newBufferWithLength:img_shape.size*sizeof(int) options:MTLResourceStorageModeShared];
    
    PPMImageShape* shapeBufferPtr = _mBufferShape.contents;
    
    shapeBufferPtr->x       = img_shape.x;
    shapeBufferPtr->y       = img_shape.y;
    shapeBufferPtr->size    = img_shape.size;
    shapeBufferPtr->memory  = img_shape.memory;
}

- (void) writeImageFromBufferToPath:(NSString *)path
{
    const char* img_path = [path UTF8String];
    
    PPMImage* img = (PPMImage*) malloc(sizeof(PPMImage));
    
    img->x = img_shape.x;
    img->y = img_shape.y;
    img->data = _mBufferResult.contents;
    
    writePPM(img_path, img);
    
    //img->data will be released automatically
    free(img);
}

- (void) printDebugBufferFrom:(int) start to:(int) end
{
    for(int i=start; i<end; i++)
    {
        printf("%d\n", ((int*)_mBufferDebug.contents)[i]);
    }
}

- (void) initializeSmoothingKernel5x5
{
    _mBufferKernel  = [_mDevice newBufferWithLength:sizeof(float)*9*9 options:MTLResourceStorageModeShared];
    
    float*  kernelPtr = _mBufferKernel.contents;
    
    for(int i=0; i<9*9; i++)
    {
        kernelPtr[i] = 1./81.;
    }
}

- (void) sendComputeCommand
{
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodeConvolutionCommand:computeEncoder];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    [commandBuffer commit];

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];
    
    //[self verifyResults];
}

- (void)encodeConvolutionCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    [computeEncoder setBuffer:_mBufferImg offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferKernel offset:0 atIndex:2];
    [computeEncoder setBuffer: _mBufferShape offset:0 atIndex: 3];
    [computeEncoder setBuffer:_mBufferDebug offset:0 atIndex:4];

    MTLSize gridSize = MTLSizeMake(img_shape.size, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    
    if (threadGroupSize > img_shape.size)
    {
        threadGroupSize = img_shape.size;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}
@end
