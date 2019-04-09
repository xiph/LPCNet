/* Copyright (c) 2017-2018 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include "celt_lpc.h"
#include <assert.h>
#include "lpcnet.h"
#include "lpcnet_private.h"
#include "opus.h"

float preemph_offset[NB_BANDS] = {1.772676, 2.937053, 0.278042, 0.299267, 0.126341, 0.060082, 0.019509, -0.017281, 0.000530, -0.000156, -0.007375, -0.010533, -0.002903, -0.005244, -0.003251, -0.000492, -0.000174, -0.004998};

void compute_band_energy_from_lpc(float *bandE, float g, const float *lpc) {
  int i;
  float sum[NB_BANDS] = {0};
  float x[WINDOW_SIZE];
  kiss_fft_cpx X[FREQ_SIZE];
  {
      RNN_CLEAR(x, WINDOW_SIZE);
      x[0] = 1;
      //x[1] = -PREEMPHASIS;
      for (i=0;i<LPC_ORDER;i++) x[i+1] = -lpc[i];
      forward_transform(X, x);
  }
#if 0
  for (i=0;i<FREQ_SIZE;i++) {
      float E = SQUARE(X[i].r) + SQUARE(X[i].i);
      printf("%g ", 1.f/(1e-15+E));
  }
  printf("\n");
#endif
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])*WINDOW_SIZE_5MS;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]*WINDOW_SIZE_5MS) + j].r);
      tmp += SQUARE(X[(eband5ms[i]*WINDOW_SIZE_5MS) + j].i);
      tmp = 1.f/(tmp + 1e-9);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
  for (i=0;i<NB_BANDS;i++) bandE[i] *= .2*g*g*(1.f/((float)WINDOW_SIZE*WINDOW_SIZE*WINDOW_SIZE*WINDOW_SIZE));
}


static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

void compute_noise(int *noise, float noise_std) {
  int i;
  for (i=0;i<FRAME_SIZE;i++) {
    noise[i] = (int)floor(.5 + noise_std*.707*(log_approx((float)rand()/RAND_MAX)-log_approx((float)rand()/RAND_MAX)));
  }
}


void write_audio(LPCNetEncState *st, const short *pcm, const int *noise, FILE *file) {
  int i, k;
  for (k=0;k<4;k++) {
  unsigned char data[4*FRAME_SIZE];
  for (i=0;i<FRAME_SIZE;i++) {
    float p=0;
    float e;
    int j;
    for (j=0;j<LPC_ORDER;j++) p -= st->features[k][2*NB_BANDS+3+j]*st->sig_mem[j];
    //printf("%f\n", pcm[k*FRAME_SIZE+i] - p);
    e = lin2ulaw(pcm[k*FRAME_SIZE+i] - p);
    /* Signal. */
    data[4*i] = lin2ulaw(st->sig_mem[0]);
    /* Prediction. */
    data[4*i+1] = lin2ulaw(p);
    /* Excitation in. */
    data[4*i+2] = st->exc_mem;
    /* Excitation out. */
    data[4*i+3] = e;
    /* Simulate error on excitation. */
    e += noise[k*FRAME_SIZE+i];
    e = IMIN(255, IMAX(0, e));
    
    RNN_MOVE(&st->sig_mem[1], &st->sig_mem[0], LPC_ORDER-1);
    st->sig_mem[0] = p + ulaw2lin(e);
    st->exc_mem = e;
  }
  fwrite(data, 4*FRAME_SIZE, 1, file);
  }
}

static short float2short(float x)
{
  int i;
  i = (int)floor(.5+x);
  return IMAX(-32767, IMIN(32767, i));
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_resp_x[2]={0};
  float mem_preemph=0;
  float x[FRAME_SIZE];
  int gain_change_count=0;
  FILE *f1;
  FILE *ffeat;
  FILE *fpcm=NULL;
  short pcm[FRAME_SIZE]={0};
  short pcmbuf[FRAME_SIZE*4]={0};
  float xbuf[FRAME_SIZE*4]={0};
  int noisebuf[FRAME_SIZE*4]={0};
  short tmp[FRAME_SIZE] = {0};
  float savedX[FRAME_SIZE] = {0};
  float speech_gain=1;
  int last_silent = 1;
  float old_speech_gain = 1;
  int one_pass_completed = 0;
  LPCNetEncState *st;
  float noise_std=0;
  int training = -1;
  int encode = 0;
  int decode = 0;
  int delay = TRAINING_OFFSET;
  int quantize = 0;
  OpusEncoder *enc;
  OpusDecoder *dec;
  enc = opus_encoder_create(16000, 1, OPUS_APPLICATION_VOIP, NULL);
  opus_encoder_ctl(enc, OPUS_SET_BITRATE(6000));
  opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_WIDEBAND));
  opus_encoder_ctl(enc, OPUS_GET_LOOKAHEAD(&delay));
  delay = 92+40;
  fprintf(stderr, "delay is %d\n", delay);
  dec = opus_decoder_create(16000, 1, NULL);
  st = lpcnet_encoder_create();
  if (argc == 5 && strcmp(argv[1], "-train")==0) training = 1;
  if (argc == 5 && strcmp(argv[1], "-qtrain")==0) {
      training = 1;
      quantize = 1;
  }
  if (argc == 4 && strcmp(argv[1], "-test")==0) training = 0;
  if (argc == 4 && strcmp(argv[1], "-qtest")==0) {
      training = 0;
      quantize = 1;
  }
  if (argc == 4 && strcmp(argv[1], "-encode")==0) {
      training = 0;
      quantize = 1;
      encode = 1;
  }
  if (argc == 4 && strcmp(argv[1], "-decode")==0) {
      training = 0;
      decode = 1;
  }
  if (training == -1) {
    fprintf(stderr, "usage: %s -train <speech> <features out> <pcm out>\n", argv[0]);
    fprintf(stderr, "  or   %s -test <speech> <features out>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[2], "r");
  if (f1 == NULL) {
    fprintf(stderr,"Error opening input .s16 16kHz speech input file: %s\n", argv[2]);
    exit(1);
  }
  ffeat = fopen(argv[3], "w");
  if (ffeat == NULL) {
    fprintf(stderr,"Error opening output feature file: %s\n", argv[3]);
    exit(1);
  }
  if (decode) {
    float vq_mem[NB_BANDS] = {0};
    while (1) {
      int ret;
      unsigned char buf[8];
      float features[4][NB_TOTAL_FEATURES];
      //int c0_id, main_pitch, modulation, corr_id, vq_end[3], vq_mid, interp_id;
      //ret = fscanf(f1, "%d %d %d %d %d %d %d %d %d\n", &c0_id, &main_pitch, &modulation, &corr_id, &vq_end[0], &vq_end[1], &vq_end[2], &vq_mid, &interp_id);
      ret = fread(buf, 1, 8, f1);
      if (ret != 8) break;
      decode_packet(features, vq_mem, buf);
      for (i=0;i<4;i++) {
        fwrite(features[i], sizeof(float), NB_TOTAL_FEATURES, ffeat);
      }
    }
    return 0;
  }
  if (training) {
    fpcm = fopen(argv[4], "w");
    if (fpcm == NULL) {
      fprintf(stderr,"Error opening output PCM file: %s\n", argv[4]);
      exit(1);
    }
  }
  while (1) {
    float E=0;
    int silent;
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i];
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) {
      if (!training) break;
      rewind(f1);
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      one_pass_completed = 1;
    }
    for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    if (training) {
      silent = E < 5000 || (last_silent && E < 20000);
      if (!last_silent && silent) {
        for (i=0;i<FRAME_SIZE;i++) savedX[i] = x[i];
      }
      if (last_silent && !silent) {
          for (i=0;i<FRAME_SIZE;i++) {
            float f = (float)i/FRAME_SIZE;
            tmp[i] = (int)floor(.5 + f*tmp[i] + (1-f)*savedX[i]);
          }
      }
      if (last_silent) {
        last_silent = silent;
        continue;
      }
      last_silent = silent;
    }
    if (count*FRAME_SIZE_5MS>=10000000 && one_pass_completed) break;
    if (training && ++gain_change_count > 2821) {
      float tmp;
      speech_gain = pow(10., (-20+(rand()%40))/20.);
      if (rand()%20==0) speech_gain *= .01;
      if (rand()%100==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_sig, b_sig);
      tmp = (float)rand()/RAND_MAX;
      noise_std = 4*tmp*tmp;
    }
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) {
      float g;
      float f = (float)i/FRAME_SIZE;
      g = f*speech_gain + (1-f)*old_speech_gain;
      x[i] *= g;
    }
    for (i=0;i<FRAME_SIZE;i++)
        xbuf[st->pcount*FRAME_SIZE + i] = (1.f/32768.f)*x[i];
    preemphasis(x, &mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) x[i] += rand()/(float)RAND_MAX - .5;
    /* PCM is delayed by 1/2 frame to make the features centered on the frames. */
    for (i=0;i<FRAME_SIZE-delay;i++) pcm[i+delay] = float2short(x[i]);
    //compute_frame_features(st, x);

    RNN_COPY(&pcmbuf[st->pcount*FRAME_SIZE], pcm, FRAME_SIZE);
    if (st->pcount == 1 || st->pcount == 3) {
        unsigned char bytes[100];
        float pcm_dec[320];
        float data[4][19];
        float bandE[4][NB_BANDS];
        int nb_bytes;
        int nb_samples;
        int pick;
        static float mem_preemph2 = 0;
        nb_bytes = opus_encode_float(enc, &xbuf[(st->pcount-1)*FRAME_SIZE], 320, bytes, 100);
        nb_samples = opus_decode_float(dec, bytes, nb_bytes, pcm_dec, 320, 0);
        preemphasis(pcm_dec, &mem_preemph2, pcm_dec, PREEMPHASIS, 2*FRAME_SIZE);
        if (nb_samples != 320) break;
        for (i=0;i<320;i++) pcm_dec[i] *= 32768;
        st->pcount--;
        compute_frame_features(st, pcm_dec);
        st->pcount++;
        compute_frame_features(st, pcm_dec+160);
        get_fdump(data);
#if 1
        for (i=0;i<4;i++) compute_band_energy_from_lpc(bandE[i], data[i][18], data[i]);
        for (i=0;i<NB_BANDS;i++) bandE[0][i] = log10(1e-2+bandE[0][i]+bandE[1][i]);
        for (i=0;i<NB_BANDS;i++) bandE[2][i] = log10(1e-2+bandE[2][i]+bandE[3][i]);
        dct(&st->features[st->pcount-1][NB_BANDS], bandE[0]);
        dct(&st->features[st->pcount][NB_BANDS]  , bandE[2]);
        st->features[st->pcount-1][NB_BANDS] -= 4;
        st->features[st->pcount][NB_BANDS] -= 4;
#endif
        pick = data[0][17] > data[1][17] ? 0 : 1;
        st->features[st->pcount-1][36] = .02*(data[pick][16] - 100);
        st->features[st->pcount-1][37] = data[pick][17] - .5;
        pick = data[2][17] > data[3][17] ? 2 : 3;
        st->features[st->pcount][36] = .02*(data[pick][16] - 100);
        st->features[st->pcount][37] = data[pick][17] - .5;

        for (i=0;i<16;i++) st->features[st->pcount-1][39+i] = -data[0][i];
        for (i=0;i<16;i++) st->features[st->pcount][39+i] = -data[2][i];

        //lpc_from_cepstrum(&st->features[st->pcount-1][2*NB_BANDS+3], st->features[st->pcount-1]);
        //lpc_from_cepstrum(&st->features[st->pcount][2*NB_BANDS+3], st->features[st->pcount]);
        //for (i=0;i<55;i++) printf("%f ", st->features[st->pcount-1][i]);
        //for (i=0;i<55;i++) printf("%f ", st->features[st->pcount][i]);
        //printf("\n");
        //printf("%f %f %f %f %f\n", st->features[st->pcount-1][37], data[1][16], data[3][16], 100+50*st->features[st->pcount-1][36], 100+50*st->features[st->pcount][36]);
    }
    if (fpcm) {
        compute_noise(&noisebuf[st->pcount*FRAME_SIZE], noise_std);
    }
    st->pcount++;
    /* Running on groups of 4 frames. */
    if (st->pcount == 4) {
#if 0
      unsigned char buf[8];
      process_superframe(st, buf, ffeat, encode, quantize);
#else
      float ftemp[55];
      static float fmem[55] = {0};
      static float last_pitch = 0;
      for (i=3;i>=0;i--) {
        if (st->features[i][36] > -1.99) last_pitch = st->features[i][36];
        else st->features[i][36] = last_pitch;
      }
      last_pitch = st->features[3][36];
#if 0
      RNN_COPY(ftemp, &st->features[3][0], 55);
      for (i=3;i>=1;i--) {
          RNN_COPY(&st->features[i][NB_BANDS], &st->features[i-1][NB_BANDS], NB_BANDS+2);
      }
      RNN_COPY(&st->features[0][NB_BANDS], &fmem[NB_BANDS], NB_BANDS+2);
      RNN_COPY(fmem, ftemp, 55);
#endif
      for (i=0;i<4;i++) {
          int j;
          for (j=0;j<NB_BANDS;j++) st->features[i][NB_BANDS+j] -= st->features[i][j];
      }
      if (ffeat) {
        for (i=0;i<4;i++) {
          fwrite(st->features[i], sizeof(float), NB_TOTAL_FEATURES, ffeat);
        }
    }
#endif
    if (fpcm) write_audio(st, pcmbuf, noisebuf, fpcm);
      st->pcount = 0;
    }
    //if (fpcm) fwrite(pcm, sizeof(short), FRAME_SIZE, fpcm);
    for (i=0;i<delay;i++) pcm[i] = float2short(x[i+FRAME_SIZE-delay]);
    old_speech_gain = speech_gain;
    count++;
  }
  fclose(f1);
  fclose(ffeat);
  if (fpcm) fclose(fpcm);
  lpcnet_encoder_destroy(st);
  return 0;
}

