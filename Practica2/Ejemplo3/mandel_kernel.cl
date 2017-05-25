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
	int i, j, iter, min, max;
	rgb_t pixel;
	double x, y, zx, zy, zx2, zy2;
	
	min = max_iter; max = 0;
	
	i = get_global_id(0);
	j = get_global_id(1);
	pixel = tex[i * width + j];
	y = (i - height/2) * scale + cy;
	x = (j - width/2) * scale + cx;
	zx = zy = zx2 = zy2 = 0;
	for(iter = 0; iter < max_iter; iter++){
		zy=2*zx*zy + y;
		zx=zx2-zy2 + x;
		zx2=zx*zx;
		zy2=zy*zy;
		if (zx2+zy2>max_iter)
			break;
	}
	if(iter < min) min = iter;
	if(iter > max) max = iter;

//	hsv_to_rgb_kernel(iter, min, max, &pixel, invert, sat, color_rotate);

	if (min == max) max = min + 1;
	if (invert) iter = max - (iter - min);
	if (!sat) {
		pixel.r = pixel.g = pixel.b = 255 * (max - iter) / (max - min);
	}
	else
	{
		double h = fmod(color_rotate + 1e-4 + 4.0 * (iter - min) / (max - min), 6);
	#	define VAL 255
		double c = VAL * sat;
		double X = c * (1 - fabs(fmod(h, 2) - 1));
	 
		pixel.r = pixel.g = pixel.b = 0;
	 
		switch((int)h) {
		case 0: pixel.r = c; pixel.g = X; break;
		case 1: pixel.r = X; pixel.g = c; break;
		case 2: pixel.g = c; pixel.b = X; break;
		case 3: pixel.g = X; pixel.b = c; break;
		case 4: pixel.r = X; pixel.b = c; break;
		default:pixel.r = c; pixel.b = X;
		}
	}

	tex[i] = pixel;
}
