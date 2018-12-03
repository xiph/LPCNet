/* NEON support for ARM machines */

#include <arm_neon.h>

static float celt_exp2(float x)
{
   int integer;
   float frac;
   union {
      float f;
      opus_uint32 i;
   } res;
   integer = floor(x);
   if (integer < -50)
      return 0;
   frac = x-integer;
   /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
   res.f = 0.99992522f + frac * (0.69583354f
           + frac * (0.22606716f + 0.078024523f*frac));
   res.i = (res.i + (integer<<23)) & 0x7fffffff;
   return res.f;
}
#define celt_exp_neon(x) celt_exp2((x)*1.44269504f)

static float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5f + .5f*tansig_approx(.5f*x);
}

static void softmax(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
        y[i] = celt_exp_neon(x[i]);
}

static void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
    {
        y[i] = tansig_approx(x[i]);
    }
}

static void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
    {
        y[i] = sigmoid_approx(x[i]);
    }
}

static void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      for (j=0;j<cols;j++)
      {
         const float * restrict w;
         float * restrict y;
         float xj;
         w = &weights[j*col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0]*xj;
         y[1] += w[1]*xj;
         y[2] += w[2]*xj;
         y[3] += w[3]*xj;
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
      }
   }
}

static void sparse_sgemv_accum16(float *out, const float *w, int rows, const int *idx, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      int cols;
      cols = *idx++;
      float * restrict y;
      y = &out[i];

      /* keep y[0..15] in registers for duration of inner loop */
      
      float32x4_t y0_3 = vld1q_f32(&y[0]);
      float32x4_t y4_7 = vld1q_f32(&y[4]);
      float32x4_t y8_11 = vld1q_f32(&y[8]);
      float32x4_t y12_15 = vld1q_f32(&y[12]);
      
      for (j=0;j<cols;j++)
      {
         float xj= x[*idx++];
	 float32x4_t wvec;
	 
	 wvec = vld1q_f32(&w[0]); y0_3 = vmlaq_n_f32(y0_3, wvec, xj);
	 wvec = vld1q_f32(&w[4]); y4_7 = vmlaq_n_f32(y4_7, wvec, xj);
	 wvec = vld1q_f32(&w[8]); y8_11 = vmlaq_n_f32(y8_11, wvec, xj);
	 wvec = vld1q_f32(&w[12]); y12_15 = vmlaq_n_f32(y12_15, wvec, xj);
	 
         w += 16;
      }

      /* save y[0..15] back to memory */
      
      vst1q_f32(&y[0], y0_3);
      vst1q_f32(&y[4], y4_7);
      vst1q_f32(&y[8], y8_11);
      vst1q_f32(&y[12], y12_15);
      
   }
}
