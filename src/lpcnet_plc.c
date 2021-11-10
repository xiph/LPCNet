/* Copyright (c) 2021 Amazon */
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

#include "lpcnet_private.h"
#include "lpcnet.h"

LPCNET_EXPORT void lpcnet_plc_init(LPCNetPLCState *st) {
  lpcnet_init(&st->lpcnet);
  lpcnet_encoder_init(&st->enc);
  RNN_CLEAR(st->pcm, PLC_BUF_SIZE);
  st->pcm_fill = PLC_BUF_SIZE;
  st->synth_fill = 0;
  st->skip_analysis = 0;
  st->blend = 0;
}

LPCNET_EXPORT LPCNetPLCState *lpcnet_plc_create() {
  LPCNetPLCState *st;
  st = malloc(sizeof(*st));
  lpcnet_plc_init(st);
  return st;
}

LPCNET_EXPORT void lpcnet_plc_destroy(LPCNetPLCState *st) {
  free(st);
}

LPCNET_EXPORT int lpcnet_plc_update(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  short output[FRAME_SIZE];
  st->enc.pcount = 0;
  if (st->skip_analysis) {
    //fprintf(stderr, "skip update\n");
    if (st->blend) {
#if 0
      float preemph_pcm[FRAME_SIZE];
      float preemph_synth[FRAME_SIZE];
      float exc[FRAME_SIZE];
      float synth_exc[FRAME_SIZE];
      for (i=1;i<FRAME_SIZE;i++) {
        preemph_pcm[i] = pcm[i] - PREEMPHASIS*pcm[i-1];
      }
      for (i=1;i<st->synth_fill;i++) {
        preemph_synth[i] = st->synth[i] - PREEMPHASIS*st->synth[i-1];
      }
      for (i=1+LPC_ORDER;i<FRAME_SIZE;i++) {
        int j;
        exc[i] = preemph_pcm[i];
        for (j=0;j<LPC_ORDER;j++) exc[i] += preemph_pcm[i-j-1]*st->lpcnet.lpc[j];
      }
      for (i=1+LPC_ORDER;i<st->synth_fill;i++) {
        int j;
        synth_exc[i] = preemph_synth[i];
        for (j=0;j<LPC_ORDER;j++) synth_exc[i] += preemph_synth[i-j-1]*st->lpcnet.lpc[j];
      }
      for (i=1;i<1+LPC_ORDER;i++) preemph_pcm[i] = st->synth[i] - PREEMPHASIS*st->synth[i-1];
      for (i=1+LPC_ORDER;i<st->synth_fill;i++) {
        float w;
        //w = (float)i*(1.f/st->synth_fill);
        w = .5 - .5*cos(M_PI*(i-LPC_ORDER-1)/(st->synth_fill-LPC_ORDER-1));
        exc[i] = w*exc[i] + (1-w)*synth_exc[i];
      }

      for (i=1+LPC_ORDER;i<FRAME_SIZE;i++) {
        int j;
        preemph_pcm[i] = exc[i];
        for (j=0;j<LPC_ORDER;j++) preemph_pcm[i] -= preemph_pcm[i-j-1]*st->lpcnet.lpc[j];
      }
      preemph_pcm[0] = st->synth[0];
      for (i=1;i<FRAME_SIZE;i++) preemph_pcm[i] = preemph_pcm[i] + PREEMPHASIS*preemph_pcm[i-1];
      for (i=0;i<FRAME_SIZE;i++) {
        float w;
        w = .5 - .5*cos(M_PI*i/FRAME_SIZE);
        pcm[i] = (int)floor(.5 + w*pcm[i] + (1-w)*preemph_pcm[i]);
      }
#else
      for (i=0;i<st->synth_fill;i++) {
        /* FIXME: Use a better window.*/
        float w;
        //w = (float)i*(1.f/st->synth_fill);
        w = .5 - .5*cos(M_PI*i/st->synth_fill);
        pcm[i] = (int)floor(.5 + w*pcm[i] + (1-w)*st->synth[i]);
      }
#endif
      st->blend = 0;
      RNN_COPY(st->pcm, &pcm[st->synth_fill], FRAME_SIZE-st->synth_fill);
      st->pcm_fill = FRAME_SIZE-st->synth_fill;
      st->synth_fill = 0;
    } else {
      RNN_COPY(&st->pcm[st->pcm_fill], pcm, FRAME_SIZE);
      st->pcm_fill += FRAME_SIZE;
    }
    //fprintf(stderr, "fill at %d\n", st->pcm_fill);
  }
  /* Update state. */
  //fprintf(stderr, "update state\n");
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
  compute_frame_features(&st->enc, x);
  process_single_frame(&st->enc, NULL);
  if (st->skip_analysis) {
    float lpc[LPC_ORDER];
    float gru_a_condition[3*GRU_A_STATE_SIZE];
    float gru_b_condition[3*GRU_B_STATE_SIZE];
    /* FIXME: backtrack state, replace features. */
    run_frame_network(&st->lpcnet, gru_a_condition, gru_b_condition, lpc, st->enc.features[0]);
    st->skip_analysis--;
  } else {
    for (i=0;i<FRAME_SIZE;i++) st->pcm[PLC_BUF_SIZE+i] = pcm[i];
    RNN_COPY(output, &st->pcm[0], FRAME_SIZE);
    lpcnet_synthesize_impl(&st->lpcnet, st->enc.features[0], output, FRAME_SIZE, FRAME_SIZE);

    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
  }
  RNN_COPY(st->features, st->enc.features[0], NB_TOTAL_FEATURES);
  st->synth_fill = 0;
  return 0;
}

LPCNET_EXPORT int lpcnet_plc_conceal(LPCNetPLCState *st, short *pcm) {
  short output[FRAME_SIZE];
  st->enc.pcount = 0;
  /* FIXME: Copy/predict features. */
  while (st->pcm_fill > 0) {
    //fprintf(stderr, "update state for PLC %d\n", st->pcm_fill);
    int update_count;
    update_count = IMIN(st->pcm_fill, FRAME_SIZE);
    RNN_COPY(output, &st->pcm[0], update_count);

    lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], output, FRAME_SIZE, update_count);
    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
    st->pcm_fill -= update_count;
    RNN_COPY(st->synth, &output[update_count], FRAME_SIZE-update_count);
    st->synth_fill = FRAME_SIZE-update_count;
    st->skip_analysis++;
  }
  //fprintf(stderr, "conceal\n");
  RNN_COPY(pcm, st->synth, st->synth_fill);
  lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], output, FRAME_SIZE, 0);
  RNN_COPY(&pcm[st->synth_fill], output, FRAME_SIZE-st->synth_fill);
  RNN_COPY(st->synth, &output[FRAME_SIZE-st->synth_fill], st->synth_fill);
  {
    int i;
    float x[FRAME_SIZE];
    /* FIXME: Can we do better? */
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  st->blend = 1;
  return 0;
}
