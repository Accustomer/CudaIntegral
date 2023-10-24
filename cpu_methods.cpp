#include "cpu_methods.h"
#include <immintrin.h>
#include <memory>
#include <cstring>


#define NUM_THREADS 8


void cpuIntegral(unsigned char* src, int* dst, const int h, const int w)
{
	dst[0] = 0;
	int i = 0, j = 0, sidx = 0, curr_didx = 0, next_didx = 0;

	// Cumulative by row
	for (i = 0; i < h; i++)
	{
		sidx = i * w;
		curr_didx = (i + 1) * (w + 1);
		next_didx = curr_didx + 1;

		// First col
		dst[curr_didx] = 0;

		// Second to last col
		for (j = 0; j < w; j++)
		{
			dst[next_didx] = dst[curr_didx] + src[sidx];
			sidx++; 
			curr_didx = next_didx;
			next_didx++;
		}
	}

	// Cumulative by col
	for (j = 1; j <= w; j++)
	{
		curr_didx = j;
		next_didx = curr_didx + w + 1;

		// First row
		dst[curr_didx] = 0;

		// Second to last row
		for (i = 0; i < h; i++)
		{
			dst[next_didx] += dst[curr_didx];
			curr_didx = next_didx;
			next_didx += w + 1;
		}
	}
}


void cpuIntegralMT(unsigned char* src, int* dst, const int h, const int w)
{
	dst[0] = 0;

	// Cumulative by row
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < h; i++)
	{
		int sidx = i * w;
		int curr_didx = (i + 1) * (w + 1);
		int next_didx = curr_didx + 1;

		// First col
		dst[curr_didx] = 0;

		// Second to last col
		for (int j = 0; j < w; j++)
		{
			dst[next_didx] = dst[curr_didx] + src[sidx];
			sidx++;
			curr_didx = next_didx;
			next_didx++;
		}
	}

	// Cumulative by col
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int j = 1; j <= w; j++)
	{
		int curr_didx = j;
		int next_didx = curr_didx + w + 1;

		// First row
		dst[curr_didx] = 0;

		// Second to last row
		for (int i = 0; i < h; i++)
		{
			dst[next_didx] += dst[curr_didx];
			curr_didx = next_didx;
			next_didx += w + 1;
		}
	}
}


void cpuIntegralSIMD(unsigned char* src, int* dst, const int h, const int w)
{
	memset(dst, 0, (w + 1) * sizeof(int));
	int i = 0, j = 0, k = 0, a_sidx = 0, a_curr_didx = 0, a_next_didx = 0;
	__m256i A, B, sidx, didx;
	__m256i ZERO = _mm256_set1_epi32(0);
	__m256i ONE = _mm256_set1_epi32(1);

	int* siptr = (int*)&sidx;
	int* diptr = (int*)&didx;
	int* bptr = (int*)&B;

	// Cumulative by row
	for (i = 0; i <= h - 8; i+=8)
	{
		// First col
		siptr[0] = i * w;
		diptr[0] = (i + 1) * (w + 1);
		dst[diptr[0]] = 0;
		for (k = 1; k < 8; k++)
		{
			siptr[k] = siptr[k - 1] + w;
			diptr[k] = diptr[k - 1] + w + 1;
			dst[diptr[k]] = 0;
		}

		// Second -> last col
		B = ZERO;
		for (j = 0; j < w; j++)
		{
			A = _mm256_setr_epi32(src[siptr[0]], src[siptr[1]], src[siptr[2]], src[siptr[3]], src[siptr[4]], src[siptr[5]], src[siptr[6]], src[siptr[7]]);
			B = _mm256_add_epi32(A, B);

			sidx = _mm256_add_epi32(sidx, ONE);
			didx = _mm256_add_epi32(didx, ONE);

			for (k = 0; k < 8; k++)
			{
				dst[diptr[k]] = bptr[k];
			}
		}
	}

	for (; i < h; i++)
	{
		a_sidx = i * w;
		a_curr_didx = (i + 1) * (w + 1);
		a_next_didx = a_curr_didx + 1;

		// First col
		dst[a_curr_didx] = 0;

		// Second to last col
		for (j = 0; j < w; j++)
		{
			dst[a_next_didx] = dst[a_curr_didx] + src[a_sidx];
			a_sidx++;
			a_curr_didx = a_next_didx;
			a_next_didx++;
		}
	}

	// Cumulative by col
	for (j = 1; j <= w - 7; j+=8)
	{
		// First row
		a_curr_didx = j;
		B = ZERO;
		for (i = 0; i < h; i++)
		{
			a_curr_didx += w + 1;
			// A = _mm256_maskz_epi32(0xffu, dst + a_curr_didx);
			// A = _mm256_loadu_epi32(dst + a_curr_didx);
			int* aptr = (int*)&A;
			for (k = 0; k < 8; k++)
			{
				aptr[k] = dst[a_curr_didx + k];
			}
			B = _mm256_add_epi32(A, B);
			memcpy(dst + a_curr_didx, bptr, 8 * sizeof(int));
		}
	}

	for (; j <= w; j++)
	{
		a_curr_didx = j;
		a_next_didx = a_curr_didx + w + 1;

		// First row
		dst[a_curr_didx] = 0;

		// Second to last row
		for (i = 0; i < h; i++)
		{
			dst[a_next_didx] += dst[a_curr_didx];
			a_curr_didx = a_next_didx;
			a_next_didx += w + 1;
		}
	}
}

