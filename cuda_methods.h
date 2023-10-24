#pragma once
#include <iostream>
#include <cuda_runtime.h>


#define CHECK(err)		__check(err, __FILE__, __LINE__)
#define CheckMsg(msg)	__checkMsg(msg, __FILE__, __LINE__)


/* Check cuda runtime api, and print error. */
inline void __check(cudaError err, const char* file, const int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CHECK() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
}


/* Check cuda runtime api, and print error with Message. */
inline void __checkMsg(const char* msg, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CheckMsg() CUDA error: %s in file <%s>, line %i : %s.\n", msg, file, line, cudaGetErrorString(err));
		exit(-1);
	}
}


class GpuTimer
{
public:
	GpuTimer(cudaStream_t stream_ = 0) : stream(stream_)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	float read()
	{
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
private:
	cudaEvent_t start, stop;
	cudaStream_t stream;
};


/* Initialize device properties */
bool initDevice(int dev);


/* 2.1 CUDA - Set row/col as a thread */
void cuIntegral(unsigned char* src, int* dst, const int3 swhp, const int3 dwhp);


/* 2.2 CUDA - Split row/col by step 8 */
void cuIntegralByStep(unsigned char* src, int* dst, const int3 swhp, const int3 dwhp);


/* 2.3 CUDA - Binary tree "Efficient Integral Image Computation on the GPU" */
void cuIntegralByBerkin(unsigned char* src, int* dst, const int3 swhp, const int3 dwhp);
