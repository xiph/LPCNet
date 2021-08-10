#include <stdio.h>
#include <math.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "tansig_table.h"

#define LPCNET_TEST

// we need to call two versions of each functions that have the same
// name, so use #defines to temp rename them

#define celt_exp2 celt_exp2_fast
#define tansig_approx tansig_approx_fast
#define sigmoid_approx sigmoid_approx_fast
#define softmax softmax_fast
#define vec_tanh vec_tanh_fast
#define vec_sigmoid vec_sigmoid_fast
#define sgemv_accum16 sgemv_accum16_fast
#define sparse_sgemv_accum16 sparse_sgemv_accum16_fast

#if defined(__AVX__) || defined(__SSE2__)
#include "vec_avx.h"
#if defined(__AVX2__)
const char simd[]="AVX2";
#elif defined(__AVX__)
const char simd[]="AVX";
#else
const char simd[]="SSE";
#endif
#elif defined(__ARM_NEON__)
#include "vec_neon.h"
const char simd[]="NEON";
#else
const char simd[]="none";

#endif

#undef celt_exp2
#undef tansig_approx
#undef sigmoid_approx
#undef softmax
#undef vec_tanh
#undef vec_sigmoid
#undef sgemv_accum16
#undef sparse_sgemv_accum16
#define sgemv_accum8x4 sgemv_accum8x4_nosimd
#define sparse_sgemv_accum8x4 sparse_sgemv_accum8x4_nosimd
#include "vec.h"

#define ROW_STEP 16
#define ROWS     ROW_STEP*10
#define COLS     2
#define ENTRIES  2

int test_sgemv_accum16() {
    float weights[ROWS*COLS];
    float x[COLS];
    float out[ROWS], out_fast[ROWS];
    int i;

    printf("sgemv_accum16.....................: ");
    for(i=0; i<ROWS*COLS; i++) {
	weights[i] = i;
    }
    for(i=0; i<ROWS; i++) {
	out[i] = 0;
	out_fast[i] = 0;
    }
  
    for(i=0; i<COLS; i++) {
	x[i] = i+1;
    }

    sgemv_accum16(out, weights, ROWS, COLS, 1, x);
    sgemv_accum16_fast(out_fast, weights, ROWS, COLS, 1, x);

    for(i=0; i<ROWS; i++) {
	if (out[i] != out_fast[i]) {
	    printf("fail\n");
	    for(i=0; i<ROWS; i++) {
		printf("%d %f %f\n", i, out[i], out_fast[i]);
		if (out[i] != out_fast[i])
		    return 1;
	    }
	}
    }

    printf("pass\n");
    return 0;
}


int test_sparse_sgemv_accum16() {
    int rows = ROW_STEP*ENTRIES;
    int indx[] = {1,0,2,0,1};
    float w[ROW_STEP*(1+2)];
    float x[ENTRIES] = {1,2};
    float out[ROW_STEP*(1+2)], out_fast[ROW_STEP*(1+2)];
    int i;

    printf("sparse_sgemv_accum16..............: ");
    for(i=0; i<ROW_STEP*(1+2); i++) {
	w[i] = i;
	out[i] = 0;
	out_fast[i] = 0;
    }
  
    sparse_sgemv_accum16(out, w, rows, indx, x);
    sparse_sgemv_accum16_fast(out_fast, w, rows, indx, x);

    for(i=0; i<ROW_STEP*ENTRIES; i++) {
	if (out[i] != out_fast[i]) {
	    printf("fail\n");
	    for(i=0; i<ROW_STEP*ENTRIES; i++) {
		printf("%d %f %f\n", i, out[i], out_fast[i]);
		if (out[i] != out_fast[i])
		    return 1;
	    }
	}
    }

    printf("pass\n");
    return 0;
}

#ifndef DOT_PROD
#define test_sgemv_accum8x4()		0
#define test_sparse_sgemv_accum8x4()	0
#else
#define ROW_STEPa       32
#define ROWSa           (ROW_STEPa*10)
#define COLSa           4

int test_sgemv_accum8x4() {
    qweight w[ROWSa*COLSa];
    float x[COLSa];
    float out_nosimd[ROWSa], out[ROWSa];
    int i;

    printf("sgemv_accum8x4....................: ");
    for(i=0; i<ROWSa*COLSa; i++) {
	w[i] = i;
    }
    for(i=0; i<ROWSa; i++) {
	out_nosimd[i] = 0;
	out[i] = 0;
    }

    for(i=0; i<COLSa; i++) {
	x[i] = i+1;
    }

    sgemv_accum8x4_nosimd(out_nosimd, w, ROWSa, COLSa, 1, x);
    sgemv_accum8x4(out, w, ROWSa, COLSa, 1, x);

    for(i=0; i<ROWSa; i++) {
	if (out_nosimd[i] != out[i]) {
	    printf("fail\n");
	    for(i=0; i<ROWSa; i++) {
		printf("%d %f %f\n", i, out_nosimd[i], out[i]);
		if (out_nosimd[i] != out[i])
		    return 1;
	    }
	}
    }

    printf("pass\n");
    return 0;
}


int test_sparse_sgemv_accum8x4() {
    int indx[(ROWSa*COLSa/32)*(COLSa+1)], *pindx;
    qweight w[ROWSa*COLSa];
    float x[COLSa];
    float out_nosimd[ROWSa], out[ROWSa];
    int i, j;

    printf("sparse_sgemv_accum8x4.............: ");
    for(i=0; i<ROWSa*COLSa; i++) {
	w[i] = i;
    }
    for(i=0; i<ROWSa; i++) {
	out_nosimd[i] = 0;
	out[i] = 0;
    }
  
    for(i=0; i<COLSa; i++) {
	x[i] = i+1;
    }
    pindx = indx;
    for(i=0; i<(ROWSa*COLSa/32); i++) {
	*pindx++ = 1;
	for(j=0; j<COLSa; j++) *pindx++ = j;
    }

    sparse_sgemv_accum8x4_nosimd(out_nosimd, w, ROWSa, COLSa, indx, x);
    sparse_sgemv_accum8x4(out, w, ROWSa, COLSa, indx, x);

    for(i=0; i<ROWSa; i++) {
	if (out_nosimd[i] != out[i]) {
	    printf("fail\n");
	    for(i=0; i<ROWSa; i++) {
		printf("%d %f %f\n", i, out_nosimd[i], out[i]);
		if (out_nosimd[i] != out[i])
		    return 1;
	    }
	}
    }

    printf("pass\n");
    return 0;
}
#endif

int main() {
    printf("testing vector routines on SIMD: %s\n", simd);
    int test1 = test_sgemv_accum16();
    int test2 = test_sparse_sgemv_accum16();
    int test3 = test_sgemv_accum8x4();
    int test4 = test_sparse_sgemv_accum8x4();

    return test1 || test2 || test3 || test4;
}

  
