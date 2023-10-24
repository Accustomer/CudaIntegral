#include "cuda_methods.h"
#include <device_launch_parameters.h>


#define X1	256
#define X2	32
#define Y2	32


///////////////////// Algorithm 1 - Kernels //////////////////////////////

__global__ void gIntegralRow(unsigned char* src, int* dst, int w1, int h1, int p1, int p2)
{
	int tid = blockIdx.x * X1 + threadIdx.x;
	if (tid < h1)
	{
		unsigned char* sptr = src + tid * p1;
		int* dptr = dst + (tid + 1) * p2;
		int i = 0, j = 1;
		while (i < w1)
		{
			dptr[j] = dptr[i] + (int)sptr[i];
			i = j;
			j++;
		}
	}
}


__global__ void gIntegralCol(int* data, int w, int h, int p)
{
	int tid = blockIdx.x * X1 + threadIdx.x + 1;
	if (tid < w)
	{
		int last_idx = tid;
		int curr_idx = tid + p;
		for (int i = 1; i < h; i++)
		{
			data[curr_idx] += data[last_idx];
			last_idx = curr_idx;
			curr_idx += p;
		}
	}
}



///////////////////// Algorithm 2 - Kernels //////////////////////////////

__global__ void gIntegralRow0Unroll8Int(unsigned char* src, int* dst, int w1, int h1, int p1, int p2)
{
	unsigned int ix = blockIdx.x * X2 + threadIdx.x;
	unsigned int iy = blockIdx.y * Y2 + threadIdx.y;
	unsigned int bix = ix * 8;
	if (bix < w1 && iy < h1)
	{
		unsigned char* sptr = src + iy * p1;
		unsigned int sidx = bix;
		unsigned int curr_didx = (iy + 1) * p2 + bix + 1;
		unsigned int last_didx = curr_didx;

		dst[curr_didx] = (int)sptr[sidx++];
#pragma unroll
		for (int i = 0; i < 7; i++)
		{
			if (sidx < w1)
			{
				last_didx = curr_didx;
				dst[++curr_didx] = (int)sptr[sidx++] + dst[last_didx];
			}
		}
	}
}


__global__ void gIntegralRow1Unroll8Int(int* data, int width, int height, int pitch)
{
	unsigned int iy = blockIdx.x * X1 + threadIdx.x + 1;
	if (iy < height)
	{
		unsigned int idx = iy * pitch;
		unsigned int i = idx + 8;
		unsigned int j = i + 8;
		unsigned int k = 16;
		while (k < width)
		{
			data[j] += data[i];
			i = j;
			j += 8;
			k += 8;
		}
	}
}


__global__ void gIntegralRow2Unroll8Int(int* data, int width, int height, int pitch)
{
	unsigned int ix = blockIdx.x * X2 + threadIdx.x;
	unsigned int iy = blockIdx.y * Y2 + threadIdx.y + 1;
	unsigned int bix = (ix + 1) * 8 + 1;
	if (bix < width && iy < height)
	{
		int* dptr = data + iy * pitch;
		unsigned int idx = bix - 1;
		unsigned int j = bix;
		int n = min(7, width - bix);
#pragma unroll
		for (int i = 0; i < n; i++)
		{
			dptr[j++] += dptr[idx];
		}
	}
}


__global__ void gIntegralCol0Unroll8Int(int* data, int width, int height, int pitch)
{
	unsigned int ix = blockIdx.x * X2 + threadIdx.x + 1;
	unsigned int iy = blockIdx.y * Y2 + threadIdx.y;
	unsigned int biy = iy * 8 + 1;
	if (ix < width && biy < height)
	{
		unsigned int idx = biy * pitch + ix;
		unsigned int idx1 = idx + pitch;
		int n = min(7, height - 1 - biy);
#pragma unroll
		for (int i = 0; i < n; i++)
		{
			data[idx1] += data[idx];
			idx = idx1;
			idx1 += pitch;
		}
	}
}


__global__ void gIntegralCol1Unroll8Int(int* data, int width, int height, int pitch)
{
	unsigned int ix = blockIdx.x * X1 + threadIdx.x + 1;
	if (ix < width)
	{
		unsigned int step = 8 * pitch;
		unsigned int i = ix + step;
		unsigned int j = i + step;
		unsigned int k = 16;
		while (k < height)
		{
			data[j] += data[i];
			i = j;
			j += step;
			k += 8;
		}
	}
}


__global__ void gIntegralCol2Unroll8Int(int* data, int width, int height, int pitch)
{
	unsigned int ix = blockIdx.x * X2 + threadIdx.x + 1;
	unsigned int iy = blockIdx.y * Y2 + threadIdx.y;
	unsigned int biy = (iy + 1) * 8 + 1;
	if (ix < width && biy < height)
	{
		unsigned int idx = (biy - 1) * pitch + ix;
		unsigned int idx1 = idx + pitch;
		int n = min(7, height - biy);
#pragma unroll
		for (int i = 0; i < n; i++)
		{
			data[idx1] += data[idx];
			idx1 += pitch;
		}
	}
}



///////////////////// Algorithm 3 - Kernels //////////////////////////////

__global__ void gScanRow(unsigned char* src, int* dst, int w1, int h1, int p1, int p2, int n)
{
	extern __shared__ int temp[];
	int bid = blockIdx.x;

	int tdx = threadIdx.x;
	int offset = 1;
	int tdx2 = tdx + tdx;
	int tdx2p = tdx2 + 1;

	int systart = bid * p1;
	int dystart = (bid + 1) * p2;

	temp[tdx2] = tdx2 < w1 ? (int)src[systart + tdx2] : 0;
	temp[tdx2p] = tdx2p < w1 ? (int)src[systart + tdx2p] : 0;
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tdx < d)
		{
			int ai = offset * tdx2p - 1;
			int bi = offset * (tdx2p + 1) - 1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (tdx == 0)
	{
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();
		if (tdx < d)
		{
			int ai = offset * tdx2p - 1;
			int bi = offset * (tdx2p + 1) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (tdx2 < w1)
	{
		dst[dystart + tdx2] = temp[tdx2];
	}
	if (tdx2p < w1)
	{
		dst[dystart + tdx2p] = temp[tdx2p];
	}
	if (tdx2 == w1 - 1 || tdx2p == w1 - 1)
	{
		dst[dystart + w1] = temp[w1 - 1] + (int)src[systart + w1 - 1];
	}
}


__global__ void gScanCol(int* data, int w, int h, int p, int n)
{
	extern __shared__ int temp[];
	int bid = blockIdx.x;

	int tdx = threadIdx.x;
	int offset = 1;
	int tdx2 = tdx + tdx;
	int tdx2p = tdx2 + 1;

	int xo = bid + 1;

	temp[tdx2] = tdx2 < h ? data[tdx2p * p + xo] : 0;
	temp[tdx2p] = tdx2p < h ? data[(tdx2p + 1) * p + xo] : 0;
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tdx < d)
		{
			int ai = offset * tdx2p - 1;
			int bi = offset * (tdx2p + 1) - 1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (tdx == 0)
	{
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();
		if (tdx < d)
		{
			int ai = offset * tdx2p - 1;
			int bi = offset * (tdx2p + 1) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (tdx2 < h)
	{
		data[tdx2 * p + xo] = temp[tdx2];
	}
	if (tdx2p < h)
	{
		data[tdx2p * p + xo] = temp[tdx2p];
	}
	if (tdx2 == h - 1 || tdx2p == h - 1)
	{
		data[h * p + xo] += temp[h - 1];
	}
}




///////////////////// Host Functions //////////////////////////////

bool initDevice(int dev)
{
	int device_count = 0;
	CHECK(cudaGetDeviceCount(&device_count));
	if (device_count == 0)
	{
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		return false;
	}
	dev = std::max<int>(0, std::min<int>(dev, device_count - 1));
	cudaDeviceProp device_prop;
	CHECK(cudaGetDeviceProperties(&device_prop, dev));
	if (device_prop.major < 1)
	{
		fprintf(stderr, "error: device does not support CUDA.\n");
		return false;
	}
	CHECK(cudaSetDevice(dev));

	int driver_version = 0;
	int runtime_version = 0;
	CHECK(cudaDriverGetVersion(&driver_version));
	CHECK(cudaRuntimeGetVersion(&runtime_version));
	fprintf(stderr, "Using Device %d: %s, CUDA Driver Version: %d.%d, Runtime Version: %d.%d\n", dev, device_prop.name,
		driver_version / 1000, driver_version % 1000, runtime_version / 1000, runtime_version % 1000);
	return true;
}


void cuIntegral(unsigned char* src, int* dst, const int3 swhp, const int3 dwhp)
{
	dim3 block1(X1);
	dim3 grid1((swhp.y + X1 - 1) / X1);
	gIntegralRow << <grid1, block1 >> > (src, dst, swhp.x, swhp.y, swhp.z, dwhp.z);

	//cv::Mat show(dwhp.y, dwhp.x, CV_32SC1);
	//size_t spitch = dwhp.x * sizeof(int);
	//size_t dpitch = dwhp.z * sizeof(int);
	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	dim3 block2(X1);
	dim3 grid2((dwhp.x - 1 + X1 - 1) / X1);
	gIntegralCol << <grid2, block2 >> > (dst, dwhp.x, dwhp.y, dwhp.z);

	//CHECK(cudaDeviceSynchronize());
	CheckMsg("integralDirectly() execution failed!\n");
}


void cuIntegralByStep(unsigned char* src, int* dst, const int3 swhp, const int3 dwhp)
{
	dim3 block0(X2, Y2);
	dim3 grid0((swhp.x + 8 * X2 - 1) / (8 * X2), (swhp.y + Y2 - 1) / Y2);
	gIntegralRow0Unroll8Int << <grid0, block0 >> > (src, dst, swhp.x, swhp.y, swhp.z, dwhp.z);

	//cv::Mat show(dwhp.y, dwhp.x, CV_32SC1);
	//size_t spitch = dwhp.x * sizeof(int);
	//size_t dpitch = dwhp.z * sizeof(int);
	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	dim3 block1(X1);
	dim3 grid1((dwhp.y - 1 + X1 - 1) / X1);
	gIntegralRow1Unroll8Int << <grid1, block1 >> > (dst, dwhp.x, dwhp.y, dwhp.z);

	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	dim3 block2(X2, Y2);
	dim3 grid2((dwhp.x + 8 * X2 - 1) / (8 * X2), (dwhp.y + Y2 - 1) / Y2);
	gIntegralRow2Unroll8Int << <grid2, block2 >> > (dst, dwhp.x, dwhp.y, dwhp.z);

	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	dim3 block3(X2, Y2);
	dim3 grid3((dwhp.x + X2 - 1) / X2, (dwhp.y + 8 * Y2 - 1) / (8 * Y2));
	gIntegralCol0Unroll8Int << <grid3, block3 >> > (dst, dwhp.x, dwhp.y, dwhp.z);

	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	dim3 block4(X1);
	dim3 grid4((dwhp.x + X1 - 1) / X1);
	gIntegralCol1Unroll8Int << <grid4, block4 >> > (dst, dwhp.x, dwhp.y, dwhp.z);

	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	dim3 block5(X2, Y2);
	dim3 grid5((dwhp.x + X2 - 1) / X2, (dwhp.y + 8 * Y2 - 1) / (8 * Y2));
	gIntegralCol2Unroll8Int << <grid5, block5 >> > (dst, dwhp.x, dwhp.y, dwhp.z);

	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	CheckMsg("integralStep() execution failed\n");
}


void cuIntegralByBerkin(unsigned char* src, int* dst, const int3 swhp, const int3 dwhp)
{
	// Scan row
	dim3 block1((int)exp2f(ceilf(log2f(swhp.x * 0.5f))));
	dim3 grid1(swhp.y);
	const int n = block1.x * 2;
	gScanRow << <grid1, block1, n * sizeof(int) >> > (src, dst, swhp.x, swhp.y, swhp.z, dwhp.z, n);

	//cv::Mat show(dwhp.y, dwhp.x, CV_32SC1);
	//size_t spitch = dwhp.x * sizeof(int);
	//size_t dpitch = dwhp.z * sizeof(int);
	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	// Scan col
	dim3 block2((int)exp2f(ceilf(log2f((dwhp.y - 1) * 0.5f))));
	dim3 grid2(dwhp.x - 1);
	const int n2 = block2.x * 2;
	gScanCol << <grid2, block2, n2 * sizeof(int) >> > (dst, dwhp.x, dwhp.y - 1, dwhp.z, n2);

	//CHECK(cudaMemcpy2D(show.data, spitch, dst, dpitch, spitch, dwhp.y, cudaMemcpyDeviceToHost));

	CheckMsg("integralBerkin() execution failed\n");
}


