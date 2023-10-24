#pragma once



/* 1.1 CPU - Directly integral */
void cpuIntegral(unsigned char* src, int* dst, const int h, const int w);


/* 1.2 CPU - Integral by multi-threads */
void cpuIntegralMT(unsigned char* src, int* dst, const int h, const int w);


/* 1.3 CPU - Integral by SIMD */
void cpuIntegralSIMD(unsigned char* src, int* dst, const int h, const int w);


