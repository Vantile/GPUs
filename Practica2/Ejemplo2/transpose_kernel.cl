// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

__kernel
void matrixTranspose(__global float * output,
                     __global float * input,
                     const    uint    width)

{
	int row = get_global_id(0);
	int col = get_global_id(1);
	int pos = row * width + col;
	int posT = col * width + row;
	output[posT] = input[pos];
}


__kernel
void matrixTransposeLocal(__global float * output,
                          __global float * input,
                          //...,
                          const    uint    width)

{

}
