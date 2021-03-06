#define MAX_WINDOW_SIZE 5*5

void buble_sort(float array[], int size)
{
	int i, j;
	float tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j] > array[j+1]){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}

__kernel
void rmnoise(__global float* imageCL,
		__global float* imageCLOUT,
		float thredshold,
		int window_size,
		int height,
		int width)
{
	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (window_size-1)>>1;
	int i = get_global_id(0) + ws2;
	int j = get_global_id(1) + ws2;
	int ii, jj;
	if(i < height-ws2 && j < width-ws2){
		for(ii = -ws2; ii<=ws2; ii++){
			for(jj = -ws2; jj<=ws2; jj++){
				window[(ii+ws2)*window_size + jj+ws2] = imageCL[(i+ii)*width + j+jj];

				buble_sort(window, window_size*window_size);
				median = window[(window_size*window_size-1)>>1];

				float aux = (median-imageCL[i*width+j])/median;
				if(aux < 0) aux = aux * (-1);

				if(aux <= thredshold)
					imageCLOUT[i*width + j] = imageCL[i*width+j];
				else
					imageCLOUT[i*width+j] = median;
			}
		}
	}	
}
