// Conjunto de Mandelbrot en OpenCL

typedef struct {unsigned char r, g, b;} rgb_t;


void hsv_to_rgb_kernel(int hue, int min, int max, rgb_t *p, int invert, int saturation, int color_rotate)
{
	if (min == max) max = min + 1;
	if (invert) hue = max - (hue - min);
	if (!saturation) {
		p->r = p->g = p->b = 255 * (max - hue) / (max - min);
		return;
	}
	double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
#	define VAL 255
	double c = VAL * saturation;
	double X = c * (1 - fabs(fmod(h, 2) - 1));
 
	p->r = p->g = p->b = 0;
 
	switch((int)h) {
	case 0: p->r = c; p->g = X; return;
	case 1: p->r = X; p->g = c; return;
	case 2: p->g = c; p->b = X; return;
	case 3: p->g = X; p->b = c; return;
	case 4: p->r = X; p->b = c; return;
	default:p->r = c; p->b = X;
	}
}

__kernel
void calcMandelCL(__global rgb_t* tex, 
			const int width, 
			const int height, 
			const double cx,
			const double cy,
			const double scale,
			const int max_iter,
			const int invert,
			const int sat,
			const int color_rotate)
{
	double zx, zy, zx2, zy2;
	int iter, min, max;
	min = max_iter; max = 0;
	int row = get_global_id(0);
	int col = get_global_id(1);

	rgb_t pixel = tex[row*width+col];

	double y = (row - height/2) * scale + cy;
	double x = (col - width/2) * scale + cx;

	zx = zy = zx2 = zy2 = 0;
	for(iter=0; iter < max_iter; iter++) {
		zy=2*zx*zy + y;
		zx=zx2-zy2 + x;
		zx2=zx*zx;
		zy2=zy*zy;
		if (zx2+zy2>max_iter)
			break;
	}
	if (iter < min) min = iter;
	if (iter > max) max = iter;

	hsv_to_rgb_kernel(iter, min, max, &pixel, invert, sat, color_rotate);

	tex[row*width+col] = pixel;
}
