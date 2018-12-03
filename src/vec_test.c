#include <stdio.h>
#include <math.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "tansig_table.h"

#ifdef __ARM_NEON__
#define sparse_sgemv_accum16 sparse_sgemv_accum16_fast
#include "vec_neon.h"
#endif

#undef sparse_sgemv_accum16
#include "vec.h"

#define ROW_STEP 16
#define ENTRIES  2

int main() {
  int rows = ROW_STEP*ENTRIES;
  int indx[] = {1,0,2,0,1};
  float w[ROW_STEP*(1+2)];
  float x[ENTRIES] = {1,2};
  float out[ROW_STEP*(1+2)], out_fast[ROW_STEP*(1+2)];
  int i;

  for(i=0; i<ROW_STEP*(1+2); i++) {
    w[i] = i;
    out[i] = 0;
    out_fast[i] = 0;
  }
  
  sparse_sgemv_accum16(out, w, rows, indx, x);
  sparse_sgemv_accum16_fast(out_fast, w, rows, indx, x);

  for(i=0; i<ROW_STEP*ENTRIES; i++) {
    printf("%d %f %f\n", i, out[i], out_fast[i]);
    if (out[i] != out_fast[i]) {
      printf("fail\n");
      return 1;
    }
  }

  printf("pass\n");
  return 0;
}
