/*************************************************************************\
*   This program is free software: you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation, either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
*                                                                         *
*   cudart (c) 2008 Erik Entrich                                          *
\*************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>

struct vector3d
{
	float x, y, z;
};

struct rgbcolor
{
	float r, g, b;
};

struct object
{
	int type;
	struct vector3d pos;
	float rad;
	struct vector3d n;
	struct rgbcolor c;
	float e;
	float d, g;
};

#define PI 3.14159265358979323846f

#define TYPE_SPHERE 1
#define TYPE_PLANE 2

#define THREADCOUNT 256
#define BLOCKCOUNT 16
//주의: 이미지 사이즈로 정확히 나눠떨어져야함!!!!!
#define TILE_WIDTH 20
#define TILE_HEIGHT 20
#define RAYTRACE_SAMPLES_PER_KERNEL_EXECUTION (THREADCOUNT * BLOCKCOUNT * 4)


struct object *objects;
struct object *device_objects;
int objectcount;
struct rgbcolor *device_imagedata;
unsigned int *device_randseeds;
vector3d *device_primary_ray_directions;

char envmap_filename[256] = "";
struct rgbcolor *envmap;
struct rgbcolor *device_envmap;
unsigned int envmap_width, envmap_height;
__device__ unsigned int device_envmap_width, device_envmap_height;
float envmap_offset;
__device__ float device_envmap_offset;

int width, height;
int numsamples;
int maxdepth;
char filename[256];
struct rgbcolor background;

__device__ int device_samples;
__device__ int device_width, device_height;
__device__ int device_maxdepth;
__device__ int device_objectcount;
__device__ struct rgbcolor device_background;

__device__ float vec_length(struct vector3d vec)
{
	return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

__device__ struct vector3d vec_normalize(struct vector3d vec)
{
	float length;
	struct vector3d vecout;

	length = vec_length(vec);

	vecout.x = vec.x / length;
	vecout.y = vec.y / length;
	vecout.z = vec.z / length;

	return vecout;
}

__device__ float vec_dot(struct vector3d veca, struct vector3d vecb)
{
	return (veca.x*vecb.x + veca.y*vecb.y + veca.z*vecb.z);
}

__device__ struct vector3d vec_cross(struct vector3d veca, struct vector3d vecb)
{
	struct vector3d vecout;

	vecout.x = veca.y*vecb.z - veca.z*vecb.y;
	vecout.y = veca.z*vecb.x - veca.x*vecb.z;
	vecout.z = veca.x*vecb.y - veca.y*vecb.x;

	return vecout;
}

__shared__ unsigned int randseed[THREADCOUNT];
__device__ float device_random(int tid)
{
#define MULTIPLIER  ((unsigned int) 1664525)
#define OFFSET  ((unsigned int) 1013904223)
#define MODULUS  ((double) 4294967296.0)
#define MODULUS_INV ((float) (1.0 / MODULUS))

	unsigned int sNew = randseed[tid] * MULTIPLIER + OFFSET;
	randseed[tid] = sNew;
	float res = sNew * MODULUS_INV;
	return res;
}

int save_pfm(unsigned int width, unsigned int height, struct rgbcolor *imagedata, char filename[], int normalize)
{
	FILE *pfm;
	int x, y;
	float r, g, b, max;

	pfm = fopen(filename, "wb");
	if (pfm != NULL)
	{
		max = 1;
		if (normalize)
		{
			for (y = 0; y<height; y++)
			for (x = 0; x<width; x++)
			{
				if (max < imagedata[x + y*width].r)
					max = imagedata[x + y*width].r;
				if (max < imagedata[x + y*width].g)
					max = imagedata[x + y*width].g;
				if (max < imagedata[x + y*width].b)
					max = imagedata[x + y*width].b;
			}
		}

		fprintf(pfm, "PF\n%i %i\n-%f\n", width, height, max);
		for (y = 0; y<height; y++)
		for (x = 0; x<width; x++)
		{
			r = imagedata[x + y*width].r / max;
			g = imagedata[x + y*width].g / max;
			b = imagedata[x + y*width].b / max;
			fwrite(&r, sizeof(float), 1, pfm);
			fwrite(&g, sizeof(float), 1, pfm);
			fwrite(&b, sizeof(float), 1, pfm);
		}

		fclose(pfm);

		return 0;
	}
	else
	{
		perror("fopen");
		return 1;
	}
}

struct rgbcolor *load_pfm(unsigned int *width, unsigned int *height, char filename[])
{
	FILE *pfm;
	char buffer[256];
	unsigned int x, y;
	float r, g, b, scale;

	struct rgbcolor *imagedata;

	pfm = fopen(filename, "rb+");
	if (pfm != NULL)
	{
		fgets(buffer, sizeof(buffer)-1, pfm);
		if (strcmp(buffer, "PF\n") != 0)
		{
			fclose(pfm);
			return NULL;
		}

		fscanf(pfm, "%u %u\n", width, height);
		fscanf(pfm, "%f\n", &scale);
		if (scale >= 0.0f)
		{
			fclose(pfm);
			return NULL;
		}

		scale *= -1.0f;

		imagedata = (struct rgbcolor*) malloc(sizeof(struct rgbcolor) * *width * *height);

		if (imagedata == NULL)
		{
			perror("malloc");
			fclose(pfm);
			return NULL;
		}

		for (y = 0; y<*height; y++)
		for (x = 0; x<*width; x++)
		{
			fread(&r, sizeof(float), 1, pfm);
			fread(&g, sizeof(float), 1, pfm);
			fread(&b, sizeof(float), 1, pfm);
			imagedata[x + y**width].r = r * scale;
			imagedata[x + y**width].g = g * scale;
			imagedata[x + y**width].b = b * scale;
		}

		fclose(pfm);
		return imagedata;
	}
	else
	{
		perror("fopen");
		return NULL;
	}
}

int build_scene(char *scenefilename)
{
	int maxobjects = 64;
	int linenum;
	char tempbuffer[64];
	FILE *scenefile;

	objectcount = 0;

	objects = new object[maxobjects];
	if (objects == NULL)
	{
		perror("malloc");
		return 0;
	}

	scenefile = fopen(scenefilename, "r");
	if (scenefile == NULL)
	{
		perror("fopen");
		return 0;
	}

	fscanf(scenefile, "%i %i %i %i\n", &width, &height, &numsamples, &maxdepth);
	fscanf(scenefile, "%f %f %f\n", &background.r, &background.g, &background.b);
	fgets(filename, 255, scenefile);
	filename[strlen(filename) - 1] = '\0';

	linenum = 3;

	while (fgets(tempbuffer, 64, scenefile) != NULL)
	{
		linenum++;

		if ((tempbuffer[0] == '#') || (tempbuffer[0] == '\n'))
			continue;

		if (strcmp(tempbuffer, "sphere\n") == 0)
		{
			objects[objectcount].type = TYPE_SPHERE;
			if (fscanf(scenefile, "%f %f %f %f\n", &objects[objectcount].pos.x, &objects[objectcount].pos.y, &objects[objectcount].pos.z, &objects[objectcount].rad) != 4)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}

			if (fscanf(scenefile, "%f %f %f\n", &objects[objectcount].c.r, &objects[objectcount].c.g, &objects[objectcount].c.b) != 3)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}

			if (fscanf(scenefile, "%f %f %f\n", &objects[objectcount].e, &objects[objectcount].d, &objects[objectcount].g) != 3)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}
		}
		else
		if (strcmp(tempbuffer, "plane\n") == 0)
		{
			objects[objectcount].type = TYPE_PLANE;
			if (fscanf(scenefile, "%f %f %f\n", &objects[objectcount].pos.x, &objects[objectcount].pos.y, &objects[objectcount].pos.z) != 3)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}

			if (fscanf(scenefile, "%f %f %f\n", &objects[objectcount].n.x, &objects[objectcount].n.y, &objects[objectcount].n.z) != 3)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}

			if (fscanf(scenefile, "%f %f %f\n", &objects[objectcount].c.r, &objects[objectcount].c.g, &objects[objectcount].c.b) != 3)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}

			if (fscanf(scenefile, "%f %f %f\n", &objects[objectcount].e, &objects[objectcount].d, &objects[objectcount].g) != 3)
			{
				fprintf(stderr, "Error in scnefile on line %i\n", linenum);
				return 0;
			}
		}
		else
		{
			fprintf(stderr, "Error in scnefile on line %i\n", linenum);
			return 0;
		}

		objectcount++;
	}

	fclose(scenefile);

	return 1;
}



__global__ void generate_primary_rays(unsigned int *randseeds, vector3d *primary_ray_directions) {

	//initialize random
	randseed[threadIdx.x] = randseeds[blockIdx.x * blockDim.x + threadIdx.x];

	__syncthreads();

	const int threadidx_global = blockIdx.x * blockDim.x + threadIdx.x;
	const int num_pixels_in_tile = TILE_WIDTH * TILE_HEIGHT;
	const int num_tiles_in_x = device_width / TILE_WIDTH;

	vector3d raydir;

	for (int sample_id = threadidx_global; sample_id < device_width * device_height * device_samples; sample_id += gridDim.x * blockDim.x) {
		int pixel_id = sample_id / device_samples;
		int tile_id = pixel_id / num_pixels_in_tile;
		int local_pixel_id = pixel_id  % num_pixels_in_tile;
		int tile_base_x = (tile_id % num_tiles_in_x) * TILE_WIDTH;
		int tile_base_y = (tile_id / num_tiles_in_x) * TILE_HEIGHT;
		int local_pixel_base_x = local_pixel_id % TILE_WIDTH;
		int local_pixel_base_y = local_pixel_id / TILE_WIDTH;

		int px = tile_base_x + local_pixel_base_x;
		int py = tile_base_y + local_pixel_base_y;

		raydir.x = ((float)px / (float)device_width) - 0.5f + device_random(threadIdx.x) / (float)device_width;
		raydir.y = (((float)py / (float)device_height) - 0.5f) * ((float)device_height / (float)device_width) + device_random(threadIdx.x) / (float)device_height;
		raydir.z = 1;
		raydir = vec_normalize(raydir);

		primary_ray_directions[sample_id] = raydir;
	}
	//propagate (changed) random seed to next kernel execution
	randseeds[blockIdx.x * blockDim.x + threadIdx.x] = randseed[threadIdx.x];
}

// need to modify this function

__global__ void raytrace(int offset, int count, struct object *globalobjects, unsigned int *randseeds, struct rgbcolor *envmap, vector3d *primary_ray_directions)
{
	volatile int px, py, tx, ty, raycounter;
	struct vector3d raydir, campos;
	struct rgbcolor pcolor, contrib;

	__shared__ struct object localobjects[64];
	volatile float mindist;
	volatile int obj, depth, counter;
	volatile float t, v, x, y, z;
	struct vector3d d, n, ir, o;

	// ************************* move global to shared : 동시에 copy 가능
	if (threadIdx.x == 0)
	{
		for (counter = 0; counter<device_objectcount; counter++)
		{
			localobjects[counter] = globalobjects[counter];
		}
	}

	randseed[threadIdx.x] = randseeds[blockIdx.x * blockDim.x + threadIdx.x];

	__syncthreads();

	const int threadidx_global = blockIdx.x * blockDim.x + threadIdx.x;

	for (int sample_id = offset + threadidx_global; sample_id < offset + count; sample_id += gridDim.x * blockDim.x) {
		pcolor.r = 0;
		pcolor.g = 0;
		pcolor.b = 0;

		// start point
		campos.x = 0;
		campos.y = 0;
		campos.z = 0;

		//ray direction calculation with pixel information.
		raydir = primary_ray_directions[sample_id];

		contrib.r = 1.0f;
		contrib.g = 1.0f;
		contrib.b = 1.0f;

		depth = 1;

		// need to check
		// one kernel/pixel --> make thread pool and get ray 
		// while 한번에 ray 하나 
		// if color is black, stop tracing
		// *************************

		while ((depth <= device_maxdepth) && ((contrib.r * 255.0f > 1.0f) || (contrib.g * 255.0f > 1.0f) || (contrib.b * 255.0f > 1.0f))) {
			mindist = 10000.0f;
			obj = -1;
			depth++;

			// get nearest object
			// intersection 이 발생할 때 그 point 계산

			for (counter = 0; counter<device_objectcount; counter++)
			{
				if (localobjects[counter].type == TYPE_SPHERE)
				{
					d.x = localobjects[counter].pos.x - campos.x;
					d.y = localobjects[counter].pos.y - campos.y;
					d.z = localobjects[counter].pos.z - campos.z;
					v = vec_dot(raydir, d);

					if (v - localobjects[counter].rad > mindist)
						continue;

					//original + direction * t = hit point 
					t = localobjects[counter].rad*localobjects[counter].rad + v*v - d.x*d.x - d.y*d.y - d.z*d.z;
					if (t < 0)
						continue;

					t = v - sqrt(t);
					if ((t > mindist) || (t < 0))
						continue;

					n.x = campos.x + t*raydir.x - localobjects[counter].pos.x;
					n.y = campos.y + t*raydir.y - localobjects[counter].pos.y;
					n.z = campos.z + t*raydir.z - localobjects[counter].pos.z;
					n = vec_normalize(n);

					mindist = t;
					obj = counter;
				}

				else if (localobjects[counter].type == TYPE_PLANE)
				{
					v = vec_dot(localobjects[counter].n, raydir);

					if (v >= 0)
						continue;

					d.x = localobjects[counter].pos.x - campos.x;
					d.y = localobjects[counter].pos.y - campos.y;
					d.z = localobjects[counter].pos.z - campos.z;

					t = vec_dot(localobjects[counter].n, d) / v;
					if ((t > mindist) || (t < 0))
						continue;

					n = localobjects[counter].n;

					mindist = t;
					obj = counter;
				}

			}

			if (obj != -1)
			{
				x = campos.x + mindist*raydir.x;
				y = campos.y + mindist*raydir.y;
				z = campos.z + mindist*raydir.z;

				ir.x = -raydir.x;
				ir.y = -raydir.y;
				ir.z = -raydir.z;

				t = 2 * vec_dot(ir, n);

				raydir.x = t*n.x - ir.x;
				raydir.y = t*n.y - ir.y;
				raydir.z = t*n.z - ir.z;
				raydir = vec_normalize(raydir);

				do
				{
					o.x = (2.0f*device_random(threadIdx.x)) - 1.0f;
					o.y = (2.0f*device_random(threadIdx.x)) - 1.0f;
					o.z = (2.0f*device_random(threadIdx.x)) - 1.0f;
				} while ((o.x*o.x + o.y*o.y + o.z*o.z > 1) || (vec_dot(o, n) <= 0));

				v = (localobjects[obj].d*device_random(threadIdx.x))*localobjects[obj].g + localobjects[obj].d*(1.0f - localobjects[obj].g);

				raydir.x = o.x*v + raydir.x*(1.0f - v);
				raydir.y = o.y*v + raydir.y*(1.0f - v);
				raydir.z = o.z*v + raydir.z*(1.0f - v);
				raydir = vec_normalize(raydir);

				campos.x = x + 0.001f*raydir.x;
				campos.y = y + 0.001f*raydir.y;
				campos.z = z + 0.001f*raydir.z;

				contrib.r *= localobjects[obj].c.r;
				contrib.g *= localobjects[obj].c.g;
				contrib.b *= localobjects[obj].c.b;

				pcolor.r += localobjects[obj].e * contrib.r;
				pcolor.g += localobjects[obj].e * contrib.g;
				pcolor.b += localobjects[obj].e * contrib.b;
			}
			else // 만난 obj 가 없을 때 : 배경으로!
			{
				if (envmap == NULL) //envmap 없을 때: 배경사진
				{
					pcolor.r += device_background.r * contrib.r;
					pcolor.g += device_background.g * contrib.g;
					pcolor.b += device_background.b * contrib.b;
				}
				else // 있을 때
				{
					//배경 이미지의 xy point
					tx = (int)(((atan2(raydir.x, raydir.z) / PI + 1.0f) * 0.5f) * device_envmap_width) % device_envmap_width;
					ty = (atan2(raydir.y, sqrt(raydir.x*raydir.x + raydir.z*raydir.z)) / PI + 0.5f) / (1.0f - device_envmap_offset) * device_envmap_height;

					if (ty >= device_envmap_height)
					{
						ty = device_envmap_height - 1;
					}

					//배경 색에 contribution: 빛이 비쳐지는 정도 곱함.
					pcolor.r += ((float)envmap[tx + ty*device_envmap_width].r) * contrib.r;
					pcolor.g += ((float)envmap[tx + ty*device_envmap_width].g) * contrib.g;
					pcolor.b += ((float)envmap[tx + ty*device_envmap_width].b) * contrib.b;
				}

				contrib.r = 0.0f;
				contrib.g = 0.0f;
				contrib.b = 0.0f;
			}
		}

		primary_ray_directions[sample_id].x = pcolor.r;
		primary_ray_directions[sample_id].y = pcolor.g;
		primary_ray_directions[sample_id].z = pcolor.b;
	}

	//pcolor.r = pcolor.r / (float)device_samples;
	//pcolor.g = pcolor.g / (float)device_samples;
	//pcolor.b = pcolor.b / (float)device_samples;

	///*	imagedata[px+py*device_width].r = 1.0f - exp( -pcolor.r );
	//imagedata[px+py*device_width].g = 1.0f - exp( -pcolor.g );
	//imagedata[px+py*device_width].b = 1.0f - exp( -pcolor.b );*/
	//imagedata[px + py*device_width].r = pcolor.r;
	//imagedata[px + py*device_width].g = pcolor.g;
	//imagedata[px + py*device_width].b = pcolor.b;

	randseeds[blockIdx.x * blockDim.x + threadIdx.x] = randseed[threadIdx.x];
}

__global__ void reconstruct(struct rgbcolor *imagedata, vector3d *primary_ray_directions) {
	const int threadidx_global = blockIdx.x * blockDim.x + threadIdx.x;
	const int num_pixels_in_tile = TILE_WIDTH * TILE_HEIGHT;
	const int num_tiles_in_x = device_width / TILE_WIDTH;
	rgbcolor pcolor;

	for (int pixel_id = threadidx_global; pixel_id < device_width * device_height; pixel_id += gridDim.x * blockDim.x) {
		int tile_id = pixel_id / num_pixels_in_tile;
		int local_pixel_id = pixel_id  % num_pixels_in_tile;
		int tile_base_x = (tile_id % num_tiles_in_x) * TILE_WIDTH;
		int tile_base_y = (tile_id / num_tiles_in_x) * TILE_HEIGHT;
		int local_pixel_base_x = local_pixel_id % TILE_WIDTH;
		int local_pixel_base_y = local_pixel_id / TILE_WIDTH;

		int px = tile_base_x + local_pixel_base_x;
		int py = tile_base_y + local_pixel_base_y;

		pcolor.r = pcolor.g = pcolor.b = 0;

		for (int sample_id = pixel_id * device_samples; sample_id < (pixel_id + 1) * device_samples; ++sample_id) {
			pcolor.r += primary_ray_directions[sample_id].x;
			pcolor.g += primary_ray_directions[sample_id].y;
			pcolor.b += primary_ray_directions[sample_id].z;
		}

		imagedata[px + py*device_width].r = pcolor.r / device_samples;
		imagedata[px + py*device_width].g = pcolor.g / device_samples;
		imagedata[px + py*device_width].b = pcolor.b / device_samples;
	}
}


//random seed 초기화 함수 따로 분리
void initialize_randseeds(unsigned int *device_randseeds, int count) {
	unsigned int *randseeds = new unsigned int[count];
	for (int counter = 0; counter < count; counter++)
		randseeds[counter] = rand();
	if (cudaMemcpy(device_randseeds, randseeds, sizeof(unsigned int)* count, cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	delete[] randseeds;
}

void render_image(int width, int height, int samples, rgbcolor *imagedata)
{
	int starttime;
	cudaError_t error;

	puts("Allocating memory on device");
	if (cudaMalloc((void **)&device_objects, sizeof(struct object) * objectcount) != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	if (cudaMalloc((void **)&device_imagedata, sizeof(struct rgbcolor) * width * height) != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	if (cudaMalloc((void **)&device_randseeds, sizeof(unsigned int)* BLOCKCOUNT * THREADCOUNT) != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	if (cudaMalloc((void **)&device_primary_ray_directions, sizeof(vector3d)* width * height * samples) != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	if (envmap != NULL)	{
		if (cudaMalloc((void **)&device_envmap, sizeof(struct rgbcolor) * envmap_width * envmap_height) != cudaSuccess)		{
			printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
		}
	}
	else {
		device_envmap = NULL;
	}

	puts("Copying data to device");
	if (cudaMemcpy(device_objects, objects, sizeof(struct object) * objectcount, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	if (cudaMemcpy(device_imagedata, imagedata, sizeof(struct rgbcolor) * width * height, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	if (envmap != NULL)
	{
		if (cudaMemcpy(device_envmap, envmap, sizeof(struct rgbcolor) * envmap_width * envmap_height, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
		}
	}

	cudaMemcpyToSymbol(device_width, &width, sizeof(width), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_height, &height, sizeof(height), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_samples, &numsamples, sizeof(numsamples), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_maxdepth, &maxdepth, sizeof(maxdepth), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_objectcount, &objectcount, sizeof(objectcount), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_background, &background, sizeof(background), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_envmap_width, &envmap_width, sizeof(envmap_width), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_envmap_height, &envmap_height, sizeof(envmap_height), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_envmap_offset, &envmap_offset, sizeof(envmap_offset), 0, cudaMemcpyHostToDevice);

	starttime = time(NULL);

	initialize_randseeds(device_randseeds, BLOCKCOUNT * THREADCOUNT);

	//Step 0: Initialize device_primary_ray_directions
	puts("Generating primary rays");
	generate_primary_rays << < BLOCKCOUNT, THREADCOUNT >> >(device_randseeds, device_primary_ray_directions);
    
    int total_samples =  width * height * numsamples;

	// *************************row 처리 :  row 단위 말고 tile  : tile is rather than rows
	for (int offset = 0; offset < total_samples; offset += RAYTRACE_SAMPLES_PER_KERNEL_EXECUTION)
    {
            int count = min(RAYTRACE_SAMPLES_PER_KERNEL_EXECUTION, total_samples - offset);
            printf("Rendering sample %i ~ %i of %i\r", offset, offset + count - 1, total_samples); fflush(stdout);

            raytrace << < BLOCKCOUNT, THREADCOUNT >> > (offset, count, device_objects, device_randseeds, device_envmap, device_primary_ray_directions);

            error = cudaGetLastError();

            if (error != cudaSuccess)
            {
                    printf("Error: %s\n", cudaGetErrorString(error));
            }

            cudaThreadSynchronize();
    }

	puts("Reconstructing");
	reconstruct << <  BLOCKCOUNT, THREADCOUNT >> > (device_imagedata, device_primary_ray_directions);

	puts("\nWaiting for threads to finish");
	cudaThreadSynchronize();

	printf("Time taken: %is\n", time(NULL) - starttime);

	puts("Copying image data from device");
	if (cudaMemcpy(imagedata, device_imagedata, sizeof(struct rgbcolor) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	cudaFree(device_primary_ray_directions);
	cudaFree(device_objects);
	cudaFree(device_imagedata);
	cudaFree(device_randseeds);
	if (envmap != NULL)
		cudaFree(device_envmap);

}

int main(int argc, char *argv[])
{
	puts("Simple CUDA Ray Tracer by 50m30n3, modified for KAIST CS610 project");

	if (argc != 2) {
		fputs("USAGE: cudart scenefile\n", stderr);
		return 1;
	}

	puts("Building scene");
	if (!build_scene(argv[1]))
	{
		puts("Cannot build scene!");
		return 1;
	}

	puts("Allocating Data");
	rgbcolor *imagedata = new rgbcolor[width * height];

	envmap = NULL;
	if (strlen(envmap_filename) > 0)
	{
		puts("Loading Envmap");

		envmap = load_pfm(&envmap_width, &envmap_height, envmap_filename);
		if (envmap != NULL)
		{
			printf("Envmap %s: %ix%i pixels\n", envmap_filename, envmap_width, envmap_height);
		}
	}

	printf("Rendering %s at %ix%i with %i samples\n", filename, width, height, numsamples);

	render_image(width, height, numsamples, imagedata);

	puts("Saving image");
	save_pfm(width, height, imagedata, filename, 0);

	delete[] objects;
	delete[] imagedata;

	return 0;
}

