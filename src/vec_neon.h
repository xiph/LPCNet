/* NEON support for ARM machines */

#include <arm_neon.h>

static void sparse_sgemv_accum16(float *out, const float *w, int rows, const int *idx, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      int cols;
      cols = *idx++;
      float * restrict y;
      y = &out[i];
      float32x4_t y0_3 = vld1q_f32(&y[0]);
      for (j=0;j<cols;j++)
      {
         float xj;
         xj = x[*idx++];

	 float32x4_t w0_3 = vld1q_f32(&w[0]);
	 y0_3 = vmlaq_n_f32(y0_3, w0_3, xj);
	 
         /*
	 y[0] += w[0]*xj;
         y[1] += w[1]*xj;
         y[2] += w[2]*xj;
         y[3] += w[3]*xj;
	 */
         y[4] += w[4]*xj;
         y[5] += w[5]*xj;
         y[6] += w[6]*xj;
         y[7] += w[7]*xj;
         y[8] += w[8]*xj;
         y[9] += w[9]*xj;
         y[10] += w[10]*xj;
         y[11] += w[11]*xj;
         y[12] += w[12]*xj;
         y[13] += w[13]*xj;
         y[14] += w[14]*xj;
         y[15] += w[15]*xj;
         w += 16;
      }
      vst1q_f32(&y[0], y0_3);
      
   }
}
