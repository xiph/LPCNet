import argparse
from ftplib import parse150
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser()

parser.add_argument('weights', metavar="<weight file>", type=str, help='model weight file in hdf5 format')
parser.add_argument('--cond-size', type=int, help="conditioning size (default: 256)", default=256)
parser.add_argument('--latent-dim', type=int, help="dimension of latent space (default: 80)", default=80)

args = parser.parse_args()

# now import the heavy stuff
import tensorflow as tf
import numpy as np
from keraslayerdump import dump_conv1d_layer, dump_dense_layer, dump_gru_layer, printVector
from rdovae import new_rdovae_model

def start_header(header_fid, header_name):
    header_guard = "_" + os.path.basename(header_name)[:-2].upper() + "_H"
    header_fid.write(
f"""
#ifndef {header_guard}
#define {header_guard}

#include "nnet.h"

"""
    )

def finish_header(header_fid):
    header_fid.write(
"""
#endif

"""
    )

def start_source(source_fid, header_name, weight_file):
    source_fid.write(
f"""
/* this source file was automatically generated from weight file {weight_file} */

#include "{header_name}"

"""
    )

def finish_source(source_fid):
    pass


def dump_statistical_model(qembedding, f, fh):
    w = qembedding.weights[0].numpy()
    levels, dim = w.shape
    N = dim // 6

    print("dumping statistical model")
    quant_scales    = tf.math.softplus(w[:, : N]).numpy()
    dead_zone       = 0.05 * tf.math.softplus(w[:, N : 2 * N]).numpy()
    r               = 0.5 + 0.5 * tf.math.sigmoid(w[:, 4 * N : 5 * N]).numpy()
    theta           = tf.math.sigmoid(w[:, 5 * N : 6 * N]).numpy()
    p0              = 1 - r ** (0.5 + 0.5 * theta)

    quant_scales_q8 = np.round(quant_scales * 2**8).astype(np.int16)
    dead_zone_q10   = np.round(dead_zone * 2**10).astype(np.int16)
    r_q15           = np.round(r * 2**15).astype(np.int16)
    p0_q15          = np.round(p0 * 2**15).astype(np.int16)

    printVector(f, quant_scales_q8, 'dred_quant_scales_q8', dtype='opus_int16', static=False)
    printVector(f, dead_zone_q10, 'dred_dead_zone_q10', dtype='opus_int16', static=False)
    printVector(f, r_q15, 'dred_r_q15', dtype='opus_int16', static=False)
    printVector(f, p0_q15, 'dred_p0_q15', dtype='opus_int16', static=False)

    fh.write(
f"""
extern const float nfec_stats_quant_scales[{levels * N}];
extern const float nfec_stats_dead_zone_theta[{levels * N}];
extern const float nfec_stats_r[{levels * N}];
extern const float nfec_stats_theta[{levels * N}];

"""
    )

if __name__ == "__main__":

    model, encoder, decoder, qembedding = new_rdovae_model(20, args.latent_dim, cond_size=args.cond_size)
    model.load_weights(args.weights)




    # encoder
    encoder_dense_names = [
        'enc_dense1',
        'enc_dense3',
        'enc_dense5',
        'enc_dense7',
        'enc_dense8',
        'gdense1',
        'gdense2'
    ]

    encoder_gru_names = [
        'enc_dense2',
        'enc_dense4',
        'enc_dense6'
    ]

    encoder_conv1d_names = [
        'bits_dense'
    ]

    source_fid = open("dred_rdovae_enc_data.c", 'w')
    header_fid = open("dred_rdovae_enc_data.h", 'w')

    start_header(header_fid, "dred_rdovae_enc_data.h")
    start_source(source_fid, "dred_rdovae_enc_data.h", os.path.basename(args.weights))

    # dump GRUs
    max_rnn_neurons_enc = max(
        [
            dump_gru_layer(encoder.get_layer(name), source_fid, header_fid)
            for name in encoder_gru_names
        ]
    )

    # dump conv layers
    max_conv_inputs = max(
        [
            dump_conv1d_layer(encoder.get_layer(name), source_fid, header_fid)
            for name in encoder_conv1d_names
        ] 
    )

    # dump Dense layers
    for name in encoder_dense_names:
        layer = encoder.get_layer(name)
        dump_dense_layer(layer, source_fid, header_fid)

    # some global constants
    header_fid.write(
f"""
#include "dred_rdovae_constants.h"

#define DRED_ENC_MAX_RNN_NEURONS {max_rnn_neurons_enc}

#define DRED_ENC_MAX_CONV_INPUTS {max_conv_inputs}

"""
    )

    finish_header(header_fid)
    finish_source(source_fid)

    header_fid.close()
    source_fid.close()

    # statistical model
    source_fid = open("dred_rdovae_stats_data.c", 'w')
    header_fid = open("dred_rdovae_stats_data.h", 'w')

    start_header(header_fid, "dred_rdovae_stats_data.h")
    start_source(source_fid, "dred_rdovae_stats_data.h", os.path.basename(args.weights))

    header_fid.write(
"""

#include "opus_types.h"

"""
    )

    dump_statistical_model(qembedding, source_fid, header_fid)

    finish_header(header_fid)
    finish_source(source_fid)

    header_fid.close()
    source_fid.close()

    # decoder
    decoder_dense_names = [
        'state1',
        'state2',
        'state3',
        'dec_dense1',
        'dec_dense3',
        'dec_dense5',
        'dec_dense7',
        'dec_dense8',
        'dec_final'
    ]   

    decoder_gru_names = [
        'dec_dense2',
        'dec_dense4',
        'dec_dense6'
    ] 

    source_fid = open("dred_rdovae_dec_data.c", 'w')
    header_fid = open("dred_rdovae_dec_data.h", 'w')

    start_header(header_fid, "dred_rdovae_dec_data.h")
    start_source(source_fid, "dred_rdovae_dec_data.h", os.path.basename(args.weights))




    # dump GRUs
    max_rnn_neurons_dec = max(
        [
            dump_gru_layer(decoder.get_layer(name), source_fid, header_fid)
            for name in decoder_gru_names
        ]
    )

    # dump Dense layers
    for name in decoder_dense_names:
        layer = decoder.get_layer(name)
        dump_dense_layer(layer, source_fid, header_fid)

    # some global constants
    header_fid.write(
f"""
#include "dred_rdovae_constants.h"

#define DRED_DEC_MAX_RNN_NEURONS {max_rnn_neurons_dec}

"""
    )

    finish_header(header_fid)
    finish_source(source_fid)

    header_fid.close()
    source_fid.close()

    # common constants
    header_fid = open("dred_rdovae_constants.h", 'w')
    start_header(header_fid, "dred_rdovae_constants.h")

    header_fid.write(
f"""
#define DRED_NUM_FEATURES 20

#define DRED_LATENT_DIM {args.latent_dim}

#define DRED_NUM_QUANTIZATION_LEVELS {qembedding.weights[0].shape[0]}

#define DRED_MAX_RNN_NEURONS {max(max_rnn_neurons_enc, max_rnn_neurons_dec)}

#define DRED_MAX_CONV_INPUTS {max_conv_inputs}
"""
    )

    finish_header(header_fid)