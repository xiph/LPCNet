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
#include "plc_data.h"

LPCNET_EXPORT int lpcnet_plc_get_size() {
  return sizeof(LPCNetPLCState);
}

LPCNET_EXPORT void lpcnet_plc_init(LPCNetPLCState *st) {
  lpcnet_init(&st->lpcnet);
  lpcnet_encoder_init(&st->enc);
  RNN_CLEAR(st->pcm, PLC_BUF_SIZE);
  st->pcm_fill = PLC_BUF_SIZE;
  st->skip_analysis = 0;
  st->blend = 0;
  st->loss_count = 0;
  st->enable_blending = 1;
  st->dc_mem = 0;
  st->remove_dc = 1;
}

LPCNET_EXPORT LPCNetPLCState *lpcnet_plc_create() {
  LPCNetPLCState *st;
  st = calloc(sizeof(*st), 1);
  lpcnet_plc_init(st);
  return st;
}

LPCNET_EXPORT void lpcnet_plc_destroy(LPCNetPLCState *st) {
  free(st);
}

static void compute_plc_pred(PLCNetState *net, float *out, const float *in) {
  float zeros[3*PLC_MAX_RNN_NEURONS] = {0};
  float dense_out[PLC_DENSE1_OUT_SIZE];
  _lpcnet_compute_dense(&plc_dense1, dense_out, in);
  compute_gruB(&plc_gru1, zeros, net->plc_gru1_state, dense_out);
  compute_gruB(&plc_gru2, zeros, net->plc_gru2_state, net->plc_gru1_state);
  _lpcnet_compute_dense(&plc_out, out, net->plc_gru2_state);
}

void clear_state(LPCNetPLCState *st) {
  RNN_CLEAR(st->lpcnet.last_sig, LPC_ORDER);
  st->lpcnet.last_exc = lin2ulaw(0.f);;
  st->lpcnet.deemph_mem = 0;
  RNN_CLEAR(st->lpcnet.nnet.gru_a_state, GRU_A_STATE_SIZE);
  RNN_CLEAR(st->lpcnet.nnet.gru_b_state, GRU_B_STATE_SIZE);
}

#define DC_CONST 0.003
#define BLEND_SIZE (FRAME_SIZE-TRAINING_OFFSET)
#if 1

/* In this causal version of the code, the DNN model implemented by compute_plc_pred()
   needs to generate two feature vectors to conceal the first lost packet.*/

LPCNET_EXPORT int lpcnet_plc_update(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  short output[FRAME_SIZE];
  float plc_features[2*NB_BANDS+NB_FEATURES+1];
  short lp[FRAME_SIZE]={0};
  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      lp[i] = (int)floor(.5 + st->dc_mem);
      st->dc_mem += DC_CONST*(pcm[i] - st->dc_mem);
      pcm[i] -= lp[i];
    }
  }
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  burg_cepstral_analysis(plc_features, x);
  st->enc.pcount = 0;
  if (st->skip_analysis) {
    /*fprintf(stderr, "skip update\n");*/
    if (st->blend) {
      short tmp[FRAME_SIZE+MAX_SYNC];
      float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
      RNN_COPY(zeros, plc_features, 2*NB_BANDS);
      zeros[2*NB_BANDS+NB_FEATURES] = 1;
      compute_plc_pred(&st->plc_net, st->features, zeros);
      if (st->enable_blending) {
        LPCNetState copy;
#if 1
        short syn[FRAME_SIZE+2*MAX_SYNC];
        short pre[FRAME_SIZE];
        float maxcorr=-10;
        int delay = 0;
        int start=0;
        copy = st->lpcnet;
        lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], tmp, FRAME_SIZE+MAX_SYNC, 0);
        RNN_COPY(syn, st->syn_buf, MAX_SYNC);
        RNN_COPY(&syn[MAX_SYNC], tmp, FRAME_SIZE+MAX_SYNC);
        for (i=FRAME_SIZE+2*MAX_SYNC-1;i>=1;i--) {
          syn[i] = syn[i] - PREEMPHASIS*syn[i-1];
        }
        for (i=1;i<FRAME_SIZE;i++) {
          pre[i] = pcm[i] - PREEMPHASIS*pcm[i-1];
        }
        pre[0] = syn[0] = 0;
        //for (i=0;i<FRAME_SIZE;i++) printf("%d ", pre[i]);
        //for (i=0;i<FRAME_SIZE+2*MAX_SYNC;i++) printf("%d ", syn[i]);
        //printf("\n");
        for (i=0;i<2*MAX_SYNC;i++)
        {
          float Rxx=0, Rxy=0, Ryy=0;
          int j;
          for (j=0;j<FRAME_SIZE;j++) {
            float x, y;
            x = syn[i+j];
            y = pre[j];
            Rxx += x*x;
            Rxy += x*y;
            Ryy += y*y;
          }
          float corr = Rxy/sqrt(1e-10+Rxx*Ryy) - .002*abs(i-MAX_SYNC);
          if (corr > maxcorr) {
            maxcorr = corr;
            delay = i-MAX_SYNC;
          }
          //printf("%f ", corr);
        }
        //printf("\n");
#if 1
        if (maxcorr > .5 && abs(delay)<20)
        {
          float ratio = 1 + (float)delay/FRAME_SIZE;
          for (i=0;i<FRAME_SIZE;i++) {
            int j;
            float frac;
            float pos = ratio*i;
            j = (int)floor(pos);
            frac = pos-j;
            syn[i] = (int)floor(.5 + (1-frac)*tmp[j] + frac*tmp[j+1]);
            if (j <= FRAME_SIZE-2) {
              pre[FRAME_SIZE-1-i] = (int)floor(.5+ (1-frac)*pcm[FRAME_SIZE-1-j] + frac*pcm[FRAME_SIZE-1-j-1]);
              start = FRAME_SIZE-1-i;
            } else {
              pre[FRAME_SIZE-1-i] = 0;
            }
          }
          /*for (i=0;i<FRAME_SIZE;i++) {
            printf("%d %d %d %d ", tmp[i], pcm[i], syn[i], pre[i]);
          }
          printf("\n");*/
          //printf("%d\n", start);
          for (i=0;i<FRAME_SIZE;i++) {
            pcm[i] = pre[i];
            tmp[i] = syn[i];
          }
        }
        for (i=0;i<start;i++) pcm[i] = tmp[i];
        for (i=0;i<FRAME_SIZE-TRAINING_OFFSET-start;i++) {
          float w;
          w = .5 - .5*cos(M_PI*i/(FRAME_SIZE-TRAINING_OFFSET-start));
          pcm[start+i] = (int)floor(.5 + w*pcm[start+i] + (1-w)*tmp[start+i]);
        }
#else
        for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) {
          float w;
          w = .5 - .5*cos(M_PI*i/(FRAME_SIZE-TRAINING_OFFSET));
          pcm[i] = (int)floor(.5 + w*pcm[i] + (1-w)*tmp[i]);
        }
#endif
#else
        copy = st->lpcnet;
        lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], tmp, FRAME_SIZE-TRAINING_OFFSET, 0);
        for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) {
          float w;
          w = .5 - .5*cos(M_PI*i/(FRAME_SIZE-TRAINING_OFFSET));
          pcm[i] = (int)floor(.5 + w*pcm[i] + (1-w)*tmp[i]);
        }
#endif
        st->lpcnet = copy;
        lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], pcm, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
      } else {
        RNN_COPY(tmp, pcm, FRAME_SIZE-TRAINING_OFFSET);
        lpcnet_synthesize_tail_impl(&st->lpcnet, tmp, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
      }
      st->blend = 0;
      RNN_COPY(st->pcm, &pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
      st->pcm_fill = TRAINING_OFFSET;
    } else {
      RNN_COPY(&st->pcm[st->pcm_fill], pcm, FRAME_SIZE);
      st->pcm_fill += FRAME_SIZE;
    }
    /*fprintf(stderr, "fill at %d\n", st->pcm_fill);*/
  }
  /* Update state. */
  /*fprintf(stderr, "update state\n");*/
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
    RNN_COPY(&plc_features[2*NB_BANDS], st->enc.features[0], NB_FEATURES);
    plc_features[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(&st->plc_net, st->features, plc_features);
    for (i=0;i<FRAME_SIZE;i++) st->pcm[PLC_BUF_SIZE+i] = pcm[i];
    RNN_COPY(output, &st->pcm[0], FRAME_SIZE);
    lpcnet_synthesize_impl(&st->lpcnet, st->enc.features[0], output, FRAME_SIZE, FRAME_SIZE);
    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
  }
  st->loss_count = 0;
  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      pcm[i] += lp[i];
    }
  }
  return 0;
}

static const float att_table[10] = {0, 0,  -.2, -.2,  -.4, -.4,  -.8, -.8, -1.6, -1.6};
LPCNET_EXPORT int lpcnet_plc_conceal(LPCNetPLCState *st, short *pcm) {
  int i;
  short output[FRAME_SIZE];
  float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
  st->enc.pcount = 0;
  /* If we concealed the previous frame, finish synthesizing the rest of the samples. */
  /* FIXME: Copy/predict features. */
  while (st->pcm_fill > 0) {
    /*fprintf(stderr, "update state for PLC %d\n", st->pcm_fill);*/
    int update_count;
    update_count = IMIN(st->pcm_fill, FRAME_SIZE);
    RNN_COPY(output, &st->pcm[0], update_count);
    compute_plc_pred(&st->plc_net, st->features, zeros);
    lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], output, update_count, update_count);
    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
    st->pcm_fill -= update_count;
    st->skip_analysis++;
  }
  lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, 0);
  compute_plc_pred(&st->plc_net, st->features, zeros);
  if (st->loss_count >= 10) st->features[0] = MAX16(-10, st->features[0]+att_table[9] - 2*(st->loss_count-9));
  else st->features[0] = MAX16(-10, st->features[0]+att_table[st->loss_count]);
  //if (st->loss_count > 4) st->features[NB_FEATURES-1] = MAX16(-.5, st->features[NB_FEATURES-1]-.1*(st->loss_count-4));
  lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], &pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, 0);
  {
    int i;
    float x[FRAME_SIZE];
    /* FIXME: Can we do better? */
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  st->loss_count++;
  st->blend = 1;
  RNN_COPY(st->syn_buf, &pcm[FRAME_SIZE-MAX_SYNC], MAX_SYNC);

  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      pcm[i] += (int)floor(.5 + st->dc_mem);
    }
  }
  return 0;
}

#else

/* In this non-causal version of the code, the DNN model implemented by compute_plc_pred()
   is always called once per frame. We process audio up to the current position minus TRAINING_OFFSET. */

LPCNET_EXPORT int lpcnet_plc_update(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  short pcm_save[FRAME_SIZE];
  float plc_features[2*NB_BANDS+NB_FEATURES+1];
  short lp[FRAME_SIZE]={0};
  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      lp[i] = (int)floor(.5 + st->dc_mem);
      st->dc_mem += DC_CONST*(pcm[i] - st->dc_mem);
      pcm[i] -= lp[i];
    }
  }
  RNN_COPY(pcm_save, pcm, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  burg_cepstral_analysis(plc_features, x);
  st->enc.pcount = 0;
  if (st->loss_count > 0) {
    LPCNetState copy;
    /* Handle blending. */
    float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
    RNN_COPY(zeros, plc_features, 2*NB_BANDS);
    zeros[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(&st->plc_net, st->features, zeros);
    copy = st->lpcnet;
    lpcnet_synthesize_impl(&st->lpcnet, st->features, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, 0);
    {
      short rev[FRAME_SIZE];
      for (i=0;i<FRAME_SIZE;i++) rev[i] = pcm[FRAME_SIZE-i-1];
      clear_state(st);
      lpcnet_synthesize_tail_impl(&st->lpcnet, rev, FRAME_SIZE, FRAME_SIZE);
      lpcnet_synthesize_tail_impl(&st->lpcnet, rev, TRAINING_OFFSET, 0);
      for (i=0;i<TRAINING_OFFSET;i++) {
        float w;
        w = .5 - .5*cos(M_PI*i/(TRAINING_OFFSET));
        st->pcm[FRAME_SIZE-1-i] = (int)floor(.5 + w*st->pcm[FRAME_SIZE-1-i] + (1-w)*rev[i]);
      }
      
    }
    st->lpcnet = copy;
    lpcnet_synthesize_impl(&st->lpcnet, st->features, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, TRAINING_OFFSET);
    //clear_state(st);
    lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);

    for (i=0;i<FRAME_SIZE;i++) x[i] = st->pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
    
  }
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
  compute_frame_features(&st->enc, x);
  process_single_frame(&st->enc, NULL);
  if (st->loss_count == 0) {
    RNN_COPY(&plc_features[2*NB_BANDS], st->enc.features[0], NB_FEATURES);
    plc_features[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(&st->plc_net, st->features, plc_features);
    lpcnet_synthesize_impl(&st->lpcnet, st->enc.features[0], &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, TRAINING_OFFSET);
    lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
  }
  RNN_COPY(&pcm[FRAME_SIZE-TRAINING_OFFSET], pcm, TRAINING_OFFSET);
  RNN_COPY(pcm, &st->pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET);
  RNN_COPY(st->pcm, pcm_save, FRAME_SIZE);
  st->loss_count = 0;
  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      pcm[i] += lp[i];
    }
  }
  return 0;
}

static const float att_table[10] = {0, 0,  -.2, -.2,  -.4, -.4,  -.8, -.8, -1.6, -1.6};
LPCNET_EXPORT int lpcnet_plc_conceal(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
  st->enc.pcount = 0;

  compute_plc_pred(&st->plc_net, st->features, zeros);
  if (st->loss_count >= 10) st->features[0] = MAX16(-10, st->features[0]+att_table[9] - 2*(st->loss_count-9));
  else st->features[0] = MAX16(-10, st->features[0]+att_table[st->loss_count]);
  //if (st->loss_count > 4) st->features[NB_FEATURES-1] = MAX16(-.5, st->features[NB_FEATURES-1]-.1*(st->loss_count-4));

  if (st->loss_count == 0) {
    RNN_COPY(pcm, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
    lpcnet_synthesize_impl(&st->lpcnet, st->features, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, TRAINING_OFFSET);
    lpcnet_synthesize_tail_impl(&st->lpcnet, &pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET, 0);
  } else {
    lpcnet_synthesize_impl(&st->lpcnet, st->features, pcm, TRAINING_OFFSET, 0);
    lpcnet_synthesize_tail_impl(&st->lpcnet, &pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET, 0);

    RNN_COPY(&st->pcm[FRAME_SIZE-TRAINING_OFFSET], pcm, TRAINING_OFFSET);
    for (i=0;i<FRAME_SIZE;i++) x[i] = st->pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  RNN_COPY(st->pcm, &pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET);

  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      pcm[i] += (int)floor(.5 + st->dc_mem);
    }
  }
  st->loss_count++;
  return 0;
}

#endif
