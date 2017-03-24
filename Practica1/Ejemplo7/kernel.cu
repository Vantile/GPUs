#include <cuda.h>
#include <math.h>

#include "kernel.h"



void cannyGPU(float *im, float *image_out,
	float level,
	int height, int width)
{
	// GPU Memory
	float *imageBWGPU;	cudaMalloc((void**)&imageBWGPU, sizeof(float)*width*height);
	cudaMemcpy(imageBWGPU, im, sizeof(float)*width*height, cudaMemcpyHostToDevice);
	float *NRGPU;		cudaMalloc((void**)&NRGPU, sizeof(float)*width*height);
	float *GGPU;		cudaMalloc((void**)&GGPU, sizeof(float)*width*height);
	float *phiGPU;		cudaMalloc((void**)&phiGPU, sizeof(float)*width*height);
	float *GxGPU;		cudaMalloc((void**)&GxGPU, sizeof(float)*width*height);
	float *GyGPU;		cudaMalloc((void**)&GyGPU, sizeof(float)*width*height);
	int *pedgeGPU;		cudaMalloc((void**)&pedgeGPU, sizeof(int)*width*height);
	float *imageOUTGPU;	cudaMalloc((void**)&imageOUTGPU, sizeof(float)*width*height);
	
	dim3 dimBlock(4, 4);
	dim3 dimGrid(height-4/16, width-4/16);

	NRGPU<<<dimGrid,dimBlock>>>(height, width, NRGPU, imageBWGPU);
	

	// Copy results to output
	cudaMemcpy(image_out, imageOUTGPU, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
}

__global__ void NRGPU(int height, int width, float *NR, float *im)
{
	int i, j;
	i = blockIdx.x + ((threadIdx.x/(width-4)) + 2);
	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}

}
