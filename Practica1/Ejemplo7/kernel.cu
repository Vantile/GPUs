#include <cuda.h>
#include <math.h>

#include "kernel.h"

__global__ void NrGPU(int height, int width, float *NR, float *im)
{
	int i, j;
	i = blockIdx.x + (threadIdx.x/(width-4)) + 2;
	j = blockIdx.y + (threadIdx.y/(height-4)) + 2;
	// Noise reduction
	NR[i*width+j] =
		 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
		+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
		+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
		+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
		+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
		/159.0;
	__syncthreads();
}

__global__ void GradGPU(int height, int width, float *NR, float *Gx, float *Gy, float *G, float *phi)
{
	float PI = 3.141593;
	int i, j;
	i = blockIdx.x + (threadIdx.x/(width-4)) + 2;
	j = blockIdx.y + (threadIdx.y/(height-4)) + 2;
	// Intensity gradient of the image
	Gx[i*width+j] = 
		 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
		+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
		+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
		+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
		+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


	Gy[i*width+j] = 
		 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
		+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
		+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
		+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

	G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
	phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

	if(fabs(phi[i*width+j])<=PI/8 )
		phi[i*width+j] = 0;
	else if (fabs(phi[i*width+j])<= 3*(PI/8))
		phi[i*width+j] = 45;
	else if (fabs(phi[i*width+j]) <= 5*(PI/8))
		phi[i*width+j] = 90;
	else if (fabs(phi[i*width+j]) <= 7*(PI/8))
		phi[i*width+j] = 135;
	else phi[i*width+j] = 0;
	__syncthreads();
}

__global__ void EdgeGPU(int height, int width, float *phi, float *G, int *pedge)
{
	int i = blockIdx.x + (threadIdx.x/(width-6)) + 3;
	int j = blockIdx.y + (threadIdx.y/(height-6)) + 3;
	pedge[i*width+j] = 0;
	if(phi[i*width+j] == 0){
		if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 45) {
		if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 90) {
		if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 135) {
		if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
			pedge[i*width+j] = 1;
	}
	__syncthreads();
}

__global__ void HystGPU(int height, int width, float level, float *G, int *pedge, float *image_out)
{
	float lowthres = level/2;
	float hithres  = 2*(level);

	int ii, jj;
	int i = blockIdx.x + (threadIdx.x/(width-6)) + 3;
	int j = blockIdx.y + (threadIdx.y/(height-6)) + 3;

	if(G[i*width+j]>hithres && pedge[i*width+j])
		image_out[i*width+j] = 255;
	else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
		// check neighbours 3x3
		for (ii=-1;ii<=1; ii++)
			for (jj=-1;jj<=1; jj++)
				if (G[(i+ii)*width+j+jj]>hithres)
					image_out[i*width+j] = 255;
	__syncthreads();
}

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

	NrGPU<<<dimGrid,dimBlock>>>(height, width, NRGPU, imageBWGPU);
	GradGPU<<<dimGrid,dimBlock>>>(height, width, NRGPU, GxGPU, GyGPU, GGPU, phiGPU);
	EdgeGPU<<<dimGrid,dimBlock>>>(height, width, phiGPU, GGPU, pedgeGPU);
	HystGPU<<<dimGrid,dimBlock>>>(height, width, level, GGPU, pedgeGPU, imageOUTGPU);
	

	// Copy results to output
	cudaMemcpy(image_out, imageOUTGPU, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
}
