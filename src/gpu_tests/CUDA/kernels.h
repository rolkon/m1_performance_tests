#ifndef KERNELS_H
#define KERNELS_H
//header to define some kernel generators
//kernels chosen from https://en.wikipedia.org/wiki/Kernel_(image_processing) 

//for debug purposes
float* kernel_identity_5x5()
{
    float* arr = (float*)malloc(25*sizeof(float));

    for(int i=0; i<25; i++)
    {
        arr[i] = 0.;
        if(i==12)
            arr[i] = 1.;
    }

    return arr;
}

float* kernel_box_blur_5x5()
{
    float* arr = (float*)malloc(25*sizeof(float));
    float div = 1./25.;

    for(int i=0; i<25; i++)
    {
        arr[i] = div;
    }

    return arr;
}

float* kernel_gaussian_blur_5x5()
{
    float* arr = (float*)malloc(25*sizeof(float));
    float div = 1./256.;

    float num[] = {1., 4., 6., 4., 1.};

    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            arr[i*5+j] = div * num[i] * num[j];
        }
    }

    return arr;
}

float* kernel_box_blur_9x9()
{
    float* arr = (float*)malloc(81*sizeof(float));
    float div = 1./81.;

    for(int i=0; i<81; i++)
    {
        arr[i] = div;
    }

    return arr;
}
#endif
