#include "cpu_methods.h"
#include "cuda_methods.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


int compareDemo(const std::string& path, const int nrepeats, const int nw, const int nh);



int main(int argc, char** argv)
{
	int dev_id = 0;
	std::string path = "D:/Project/2023/CUDA-Integral/github/supplementary/sample.png";
	int repeat_times = 100;
	int new_width = -1, new_height = -1;
	if (argc > 1)
	{
		dev_id = atoi(argv[1]);
	}
	if (argc > 2)
	{
		path = argv[2];
	}
	if (argc > 3)
	{
		repeat_times = atoi(argv[3]);
	}
	if (argc > 5)
	{
		new_width = atoi(argv[4]);
		new_height = atoi(argv[5]);
	}
	int ret = initDevice(dev_id);
	ret = compareDemo(path, repeat_times, new_width, new_height);

	return ret;
}




int compareDemo(const std::string& path, const int nrepeats, const int nw, const int nh)
{
	// Read image
	cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		printf("Can not read image from %s\n", path.c_str());
		return -1;
	}

	if (nw != -1 && nh != -1)
	{
		cv::resize(image, image, cv::Size(nw, nh));
	}

	// Allocate memory On CPU
	const int h = image.rows;
	const int w = image.cols;
	cv::Mat base, cpures1, cpures2, cpures3, cures1, cures2, cures3;
	base = cv::Mat(h + 1, w + 1, CV_32SC1);
	base.copyTo(cpures1); base.copyTo(cpures2); base.copyTo(cpures3);
	base.copyTo(cures1); base.copyTo(cures2); base.copyTo(cures3);
	int* cpuptr1 = cpures1.ptr<int>(0);
	int* cpuptr2 = cpures2.ptr<int>(0);
	int* cpuptr3 = cpures3.ptr<int>(0);

	// Allocate memory On GPU
	int3 swhp{ w, h, 0 };
	unsigned char* d_src = NULL;
	size_t dpitch1 = 0;
	size_t spitch1 = sizeof(unsigned char) * swhp.x;
	CHECK(cudaMallocPitch((void**)&d_src, &dpitch1, spitch1, swhp.y));
	swhp.z = dpitch1 / sizeof(unsigned char);

	int3 dwhp{ w + 1, h + 1,0 };
	int* d_dst1 = NULL;
	int* d_dst2 = NULL;
	int* d_dst3 = NULL;
	size_t dpitch2 = 0;
	size_t spitch2 = sizeof(int) * dwhp.x;
	CHECK(cudaMallocPitch((void**)&d_dst1, &dpitch2, spitch2, dwhp.y));
	CHECK(cudaMallocPitch((void**)&d_dst2, &dpitch2, spitch2, dwhp.y));
	CHECK(cudaMallocPitch((void**)&d_dst3, &dpitch2, spitch2, dwhp.y));
	dwhp.z = dpitch2 / sizeof(int);

	CHECK(cudaMemcpy2D(d_src, dpitch1, image.data, spitch1, spitch1, swhp.y, cudaMemcpyHostToDevice));
	CHECK(cudaMemset(d_dst1, 0, dpitch2 * dwhp.y));
	CHECK(cudaMemset(d_dst2, 0, dpitch2 * dwhp.y));
	CHECK(cudaMemset(d_dst3, 0, dpitch2 * dwhp.y));

	// Warm up
	for (int i = 0; i < 10; i++)
	{
		cuIntegral(d_src, d_dst1, swhp, dwhp);
	}

	// 0 OpenCV - Base	
	int64 base_t0 = cv::getTickCount();
	for (int i = 0; i < nrepeats; i++)
	{
		cv::integral(image, base);
	}
	int64 base_elapsed = cv::getTickCount() - base_t0;

	// 1.1 CPU - Directly integral
	int64 cpu1_t0 = cv::getTickCount();
	for (int i = 0; i < nrepeats; i++)
	{
		cpuIntegral(image.data, cpuptr1, h, w);
	}
	int64 cpu1_elapsed = cv::getTickCount() - cpu1_t0;

	// 1.2 CPU - Integral by multi-threads
	int64 cpu2_t0 = cv::getTickCount();
	for (int i = 0; i < nrepeats; i++)
	{
		cpuIntegralMT(image.data, cpuptr2, h, w);
	}
	int64 cpu2_elapsed = cv::getTickCount() - cpu2_t0;

	// 1.3 CPU - Integral by SIMD
	int64 cpu3_t0 = cv::getTickCount();
	for (int i = 0; i < nrepeats; i++)
	{
		cpuIntegralSIMD(image.data, cpuptr3, h, w);
	}
	int64 cpu3_elapsed = cv::getTickCount() - cpu3_t0;

	// 2.1 CUDA - Set row/col as a thread
	GpuTimer timer(0);
	for (int i = 0; i < nrepeats; i++)
	{
		cuIntegral(d_src, d_dst1, swhp, dwhp);
	}
	float cuda1_elapsed = timer.read();

	// 2.2 CUDA - Split row/col by step 8
	float cuda2_t0 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		cuIntegralByStep(d_src, d_dst2, swhp, dwhp);
	}
	float cuda2_elapsed = timer.read() - cuda2_t0;

	// 2.3 CUDA - Binary tree "Efficient Integral Image Computation on the GPU"
	float cuda3_t0 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		cuIntegralByBerkin(d_src, d_dst3, swhp, dwhp);
	}
	float cuda3_elapsed = timer.read() - cuda3_t0;

	// Compare results
	CHECK(cudaMemcpy2D(cures1.data, spitch2, d_dst1, dpitch2, spitch2, dwhp.y, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy2D(cures2.data, spitch2, d_dst2, dpitch2, spitch2, dwhp.y, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy2D(cures3.data, spitch2, d_dst3, dpitch2, spitch2, dwhp.y, cudaMemcpyDeviceToHost));

	cv::Mat cpudiff1, cpudiff2, cpudiff3, cudiff1, cudiff2, cudiff3;
	cv::absdiff(cpures1, base, cpudiff1);
	cv::absdiff(cpures2, base, cpudiff2);
	cv::absdiff(cpures3, base, cpudiff3);
	cv::absdiff(cures1, base, cudiff1);
	cv::absdiff(cures2, base, cudiff2);
	cv::absdiff(cures3, base, cudiff3);

	double cpu_maxdiff1 = -1;
	double cpu_maxdiff2 = -1;
	double cpu_maxdiff3 = -1;
	double cu_maxdiff1 = -1;
	double cu_maxdiff2 = -1;
	double cu_maxdiff3 = -1;
	cv::minMaxLoc(cpudiff1, NULL, &cpu_maxdiff1, NULL, NULL);
	cv::minMaxLoc(cpudiff2, NULL, &cpu_maxdiff2, NULL, NULL);
	cv::minMaxLoc(cpudiff3, NULL, &cpu_maxdiff3, NULL, NULL);
	cv::minMaxLoc(cudiff1, NULL, &cu_maxdiff1, NULL, NULL);
	cv::minMaxLoc(cudiff2, NULL, &cu_maxdiff2, NULL, NULL);
	cv::minMaxLoc(cudiff3, NULL, &cu_maxdiff3, NULL, NULL);

	double freq = 1000.0 / cv::getTickFrequency() / nrepeats;
	printf("Image Size: (%d, %d)\n", w, h);
	printf("Base - OpenCV 4.7.0, Time cost: %lfms\n", base_elapsed * freq);
	printf("CPU method1, Diff: %lf, Time cost: %lfms\n", cpu_maxdiff1, cpu1_elapsed * freq);
	printf("CPU method2, Diff: %lf, Time cost: %lfms\n", cpu_maxdiff2, cpu2_elapsed * freq);
	printf("CPU method3, Diff: %lf, Time cost: %lfms\n", cpu_maxdiff3, cpu3_elapsed * freq);
	printf("CUDA method1, Diff: %lf, Time cost: %lfms\n", cu_maxdiff1, cuda1_elapsed / nrepeats);
	printf("CUDA method2, Diff: %lf, Time cost: %lfms\n", cu_maxdiff2, cuda2_elapsed / nrepeats);
	printf("CUDA method3, Diff: %lf, Time cost: %lfms\n", cu_maxdiff3, cuda3_elapsed / nrepeats);

	// Free memory
	CHECK(cudaFree(d_src));
	CHECK(cudaFree(d_dst1));
	CHECK(cudaFree(d_dst2));
	CHECK(cudaFree(d_dst3));

	return 0;
}

