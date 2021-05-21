/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalAdder.h"
#import "ppm_file_reader.h"

// The number of floats in each array, and the size of the arrays in bytes.
//const unsigned int arrayLength = 1 << 24;
//const unsigned int bufferSize = arrayLength * sizeof(float);

@implementation MetalAdder
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mAddFunctionPSO;

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;

    // Buffers to hold data.
    //id<MTLBuffer> _mBufferA;
    //id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferImg;
    id<MTLBuffer> _mBufferResult;
    
    int sizeX, sizeY;
    size_t imgSize;
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

        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"switch_color_channels"];
        if (addFunction == nil)
        {
            NSLog(@"Failed to find the adder function.");
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
    sizeX = img->x;
    sizeY = img->y;
    
    imgSize = sizeX * sizeY * sizeof(PPMPixel);
    
    _mBufferImg = [_mDevice newBufferWithBytesNoCopy:(void*)img->data length:imgSize options:MTLResourceStorageModeShared deallocator:nil];
    
    // Allocate three buffers to hold our initial data and the result.
    //_mBufferA = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    //_mBufferB = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:imgSize options:MTLResourceStorageModeShared];
    
    //[self generateRandomFloatData:_mBufferA];
    //[self generateRandomFloatData:_mBufferB];
}

- (void) writeImageFromBufferToPath:(NSString *)path
{
    const char* img_path = [path UTF8String];
    
    PPMImage* img = (PPMImage*) malloc(sizeof(PPMImage));
    
    img->x = sizeX;
    img->y = sizeY;
    img->data = _mBufferResult.contents;
    
    writePPM(img_path, img);
    
    //img->data will be released automatically
    free(img);
}

- (void) sendComputeCommand
{
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodeAddCommand:computeEncoder];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    [commandBuffer commit];

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];

    //[self verifyResults];
}

- (void)encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    //[computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    //[computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferImg offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(imgSize, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > imgSize)
    {
        threadGroupSize = imgSize;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}
/*
- (void) generateRandomFloatData: (id<MTLBuffer>) buffer
{
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < imgSize; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}
- (void) verifyResults
{
    //float* a = _mBufferA.contents;
    //float* b = _mBufferB.contents;
    char* img = _mBufferImg.contents;
    char* result = _mBufferResult.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (result[index] != (a[index] + b[index]))
        {
            printf("Compute ERROR: index=%lu result=%g vs %g=a+b\n",
                   index, result[index], a[index] + b[index]);
            assert(result[index] == (a[index] + b[index]));
        }
    }
    printf("Compute results as expected\n");
}
*/
@end
