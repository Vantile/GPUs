#include <stdio.h>
#include "my_ocl.h"
#include "CL/cl.h"


double calc_piOCL(int n)
{
	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;

	cl_int err;

	cl_mem areaCL;

	size_t global[1];

	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;

	// read the kernel
	fp = fopen("pi.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen]='\0';

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to find a platform!\n");
		//return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n");
        	//return EXIT_FAILURE;
	}

	int i;
	// Secure a GPU
	for (i = 0; i < numPlatforms; i++)
	{
     	   	err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        	if (err == CL_SUCCESS)
        	{
            		break;
       		}
   	}

	if (device_id == NULL)
	{
		printf("Error: Failed to create a device group!\n");
		//return EXIT_FAILURE;
	}
  
	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
    	{
        	printf("Error: Failed to create a compute context!\n");
        	//return EXIT_FAILURE;
   	}

	// Create a command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (!command_queue)
	{
        	printf("Error: Failed to create a command commands!\n");
        	//return EXIT_FAILURE;
	}

	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 ,(const char **)
                                          &kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create program object.");
		exit(1);
	}       
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
        	printf("Build failed.");

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, "calcpi", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object.");
		exit(1);
	}

	// create buffer objects of kernel function
	areaCL = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*n, NULL, NULL);
	if(!areaCL)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	// set the kernel arguments
	if( (clSetKernelArg(kernel, 0, sizeof(cl_mem), &areaCL)) ||
		( clSetKernelArg(kernel, 1, sizeof(int), &n)) != CL_SUCCESS)
	{
		printf("Unable to set kernel arguments. Error code=%d\n", err);
		exit(1);
	}

	global[0] = n;

	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	if(err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command. Error Code=%d\n", err);
		exit(1);
	}

	clFinish(command_queue);

	double *area; 
	area = (double *) malloc(sizeof(double)*n);
	double pi, sumArea;

	// read the output
	err = clEnqueueReadBuffer(command_queue, areaCL, CL_TRUE, 0, sizeof(double)*n,
					area, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		printf("Error enqueuing read buffer command. Error Code=%d\n", err);
		exit(1);
	}
	for(i = 1; i < n; i++)
		sumArea += area[i];
	pi = sumArea / n;

	clReleaseMemObject(areaCL);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return pi;
}
