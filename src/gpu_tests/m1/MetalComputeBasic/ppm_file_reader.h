#ifndef PPM_FILE_READER_H
#define PPM_FILE_READER_H

#include <stdlib.h>
#include <stdio.h>

//PPM code reading from
//https://stackoverflow.com/a/2699908/13768657

typedef struct {
    unsigned char red,green,blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

//dimensions x and y, size = x*y, memory = size * sizeof(PPMPixel)
typedef struct {
    unsigned int x, y;
    unsigned int size;
    unsigned int memory;
} PPMImageShape;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

//allocates 16K aligned memory for image
static PPMImage *readPPM(const char *filename)
{
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    //open PPM file for reading
    fp = fopen(filename, "rb");
    
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s': '%s'\n", filename, strerror(errno));
        exit(1);
    }
    
    //read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }
    
    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }
    
    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    
    //check for comments
    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n') ;
        c = getc(fp);
    }
    
    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }
    
    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        exit(1);
    }
    
    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }
    
    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    //make it 16K byte aligned, so it's usable as a GPU buffer for MTL
    img->data = (PPMPixel*)aligned_alloc(16384, img->x * img->y * sizeof(PPMPixel));
    
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    
    //read pixel data from file
    if (fread(img->data, sizeof(PPMPixel) * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }
    
    fclose(fp);
    return img;
}
void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }
    
    //write the header file
    //image format
    fprintf(fp, "P6\n");
    
    //comments
    //fprintf(fp, "# Created by %s\n",CREATOR);
    
    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);
    
    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);
    
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

void changeColorPPM(PPMImage *img)
{
    int i;
    if(img){
        
        for(i=0;i<img->x*img->y;i++){
            img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
            img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
            img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
        }
    }
}

#endif
