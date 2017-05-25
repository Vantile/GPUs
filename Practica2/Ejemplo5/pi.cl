

__kernel
void calcpi(__global double* areaCL, int n){
	int i = get_global_id(0);
	if(i > 0){
		double x = (i+0.5)/n;
		areaCL[i] = 4.0/(1.0 + x * x);
	}
}
