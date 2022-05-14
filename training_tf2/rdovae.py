#!/usr/bin/python3
'''Copyright (c) 2022 Amazon

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
'''

import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, GaussianNoise
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l1
import numpy as np
import h5py

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        # Ensure that abs of adjacent weights don't sum to more than 127. Otherwise there's a risk of
        # saturation when implementing dot products with SSSE3 or AVX2.
        return self.c*p/tf.maximum(self.c, tf.repeat(tf.abs(p[:, 1::2])+tf.abs(p[:, 0::2]), 2, axis=1))
        #return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

constraint = WeightClip(0.992)

rand_gen = tf.random.Generator.from_non_deterministic_state()

def bits_noise(x):
    
    return K.clip(1.1*x + .2*rand_gen.uniform((16, 1000, 50,))-.1, -1, 1)
    #return K.clip(x + 0.1*(1-K.abs(x))*GaussianNoise(1.)(x), -1, 1)

def binary_reg(c):
    def binary(x):
        #return c*K.mean(K.sqrt(1.01-K.abs(x))-.1)
        return c*K.mean(K.abs(x))
    return binary

def commit_reg(c):
    def commit(x):
        #return c*K.mean(K.sqrt(1.01-K.abs(x))-.1)
        return c*K.mean(K.abs(x-tf.round(x)))
    return commit

def soft_quantize(x):
    #x = 4*x
    x = x - (.25/np.math.pi)*tf.math.sin(2*np.math.pi*x)
    x = x - (.25/np.math.pi)*tf.math.sin(2*np.math.pi*x)    
    return x

def hard_quantize(x):
    #x = soft_quantize(x)
    quantized = tf.round(x)
    return x + tf.stop_gradient(quantized - x)

def rate_loss(y_true,y_pred):
    log2_e = 1.4427
    n = y_pred.shape[-1]
    C = n - log2_e*np.math.log(np.math.gamma(n))
    k = K.sum(K.abs(y_pred), axis=-1)
    p = 1.5
    #rate = C + (n-1)*log2_e*tf.math.log((k**p + (n/5)**p)**(1/p))
    rate = C + (n-1)*log2_e*tf.math.log(k + .112*n**2/(n/1.8+k) )
    return K.mean(rate)

eps=1e-6
def safelog2(x):
    log2_e = 1.4427
    return log2_e*tf.math.log(eps+x)

def sq_rate_loss(y_true,y_pred):
    log2_e = 1.4427
    n = y_pred.shape[-1]//3
    r = y_pred[:,:,2*n:]
    p0 = y_pred[:,:,n:2*n]
    p0 = 1-r**(.5+.5*p0)
    y_pred = y_pred[:,:,:n]
    y0 = K.maximum(0., 1. - K.abs(y_pred))**2
    rate = -y0*safelog2(p0*r**K.abs(y_pred)) - (1-y0)*safelog2(.5*(1-p0)*(1-r)*r**(K.abs(y_pred)-1))
    rate = K.sum(rate, axis=-1)
    return K.mean(rate)

def sq1_rate_loss(y_true,y_pred):
    log2_e = 1.4427
    n = y_pred.shape[-1]//3
    r = (y_pred[:,:,2*n:])
    p0 = (y_pred[:,:,n:2*n])
    p0 = 1-r**(.5+.5*p0)
    y_pred = y_pred[:,:,:n]
    y0 = K.maximum(0., 1. - K.abs(y_pred))**2
    rate = -y0*safelog2(p0*r**K.abs(y_pred)) - (1-y0)*safelog2(.5*(1-p0)*(1-r)*r**(K.abs(y_pred)-1))
    rate = -safelog2(-.5*tf.math.log(r)*r**K.abs(y_pred))
    rate = -safelog2((1-r)/(1+r)*r**K.abs(y_pred))
    rate = K.sum(rate, axis=-1)
    return K.mean(rate)

def sq2_rate_loss(y_true,y_pred):
    log2_e = 1.4427
    n = y_pred.shape[-1]//3
    r = y_pred[:,:,2*n:]
    p0 = y_pred[:,:,n:2*n]
    p0 = 1-r**(.5+.5*p0)
    y_pred = tf.round(y_pred[:,:,:n])
    y0 = K.maximum(0., 1. - K.abs(y_pred))**2
    rate = -y0*safelog2(p0*r**K.abs(y_pred)) - (1-y0)*safelog2(.5*(1-p0)*(1-r)*r**(K.abs(y_pred)-1))
    rate = K.sum(rate, axis=-1)
    return K.mean(rate)

def sq_rate_metric(y_true,y_pred):
    log2_e = 1.4427
    n = y_pred.shape[-1]//3
    r = y_pred[:,:,2*n:]
    p0 = y_pred[:,:,n:2*n]
    p0 = 1-r**(.5+.5*p0)
    y_pred = tf.round(y_pred[:,:,:n])
    #FIXME: make y0 differentiable
    #y0 = tf.cast(K.abs(y_pred)<.5, tf.float32)
    y0 = K.maximum(0., 1. - K.abs(y_pred))**2
    rate = -y0*log2_e*tf.math.log(p0) - (1-y0)*log2_e*tf.math.log(.5*(1-p0)*(1-r)*r**(K.abs(y_pred)-1))
    rate = K.sum(rate, axis=-1)
    return K.mean(rate)

def rate_metric(y_true,y_pred):
    log2_e = 1.4427
    n = y_pred.shape[-1]
    C = n - log2_e*np.math.log(np.math.gamma(n))
    k = K.sum(K.abs(tf.round(y_pred)), axis=-1)
    p = 1.5
    #rate = C + (n-1)*log2_e*tf.math.log((k**p + (n/5)**p)**(1/p))
    rate = C + (n-1)*log2_e*tf.math.log(k + .112*n**2/(n/1.8+k) )
    return K.mean(rate)

def new_rdovae_model(nb_used_features=20, nb_bits=17, batch_size=128, cond_size=128, cond_size2=128):
    feat = Input(shape=(None, nb_used_features), batch_size=batch_size)


    enc_dense1 = Dense(cond_size2, activation='tanh', name='enc_dense1')
    enc_dense2 = Dense(cond_size, activation='tanh', name='enc_dense2')
    enc_dense3 = Dense(cond_size2, activation='tanh', name='enc_dense3')
    enc_dense4 = Dense(cond_size, activation='tanh', name='enc_dense4')
    enc_dense5 = Dense(cond_size2, activation='tanh', name='enc_dense5')
    enc_dense6 = Dense(cond_size, activation='tanh', name='enc_dense6')
    enc_dense7 = Dense(cond_size, activation='tanh', name='enc_dense7')
    enc_dense8 = Dense(cond_size, activation='tanh', name='enc_dense8')
    #enc_dense6 = Bidirectional(CuDNNGRU(cond_size, return_sequences=True, name='enc_dense6'))
    #enc_dense7 = Bidirectional(CuDNNGRU(cond_size, return_sequences=True, name='enc_dense7'))
    #enc_dense8 = Bidirectional(CuDNNGRU(cond_size, return_sequences=True, name='enc_dense8'))

    #bits_dense = Dense(nb_bits, activation='linear', name='bits_dense', activity_regularizer=binary_reg(4.25))
    #bits_dense = Dense(nb_bits, activation='tanh', name='bits_dense')
    #bits_dense = Dense(nb_bits, activation='linear', name='bits_dense')
    bits_dense = Dense(nb_bits, activation='linear', name='bits_dense')
    #bits_dense = Dense(nb_bits, activation='linear', name='bits_dense', activity_regularizer=commit_reg(1.))


    #bits = bits_dense(enc_dense8(enc_dense7(enc_dense6(enc_dense5(enc_dense4(enc_dense3(enc_dense2(enc_dense1(feat)))))))))
    d1 = enc_dense1(Reshape((-1, 4*nb_used_features))(feat))
    d2 = enc_dense2(d1)
    d3 = enc_dense3(d2)
    d4 = enc_dense4(d3)
    d5 = enc_dense5(d4)
    d6 = enc_dense6(d5)
    d7 = enc_dense7(d6)
    d8 = enc_dense8(d7)
    bits = bits_dense(Concatenate()([d1, d2, d3, d4, d5, d6, d7, d8]))
    #bits = bits_dense(feat)
    #bits = enc_dense1(feat)
    encoder = Model(feat, bits, name='bits')

    bits_input = Input(shape=(None, nb_bits), batch_size=batch_size)

    noise_lambda = Lambda(bits_noise)
    #noisy_bits = noise_lambda(bits)
    
    
    dec_dense1 = Dense(cond_size2, activation='tanh', name='dec_dense1')
    dec_dense2 = Dense(cond_size, activation='tanh', name='dec_dense2')
    dec_dense3 = Dense(cond_size2, activation='tanh', name='dec_dense3')
    dec_dense4 = Dense(cond_size, activation='tanh', name='dec_dense4')
    dec_dense5 = Dense(cond_size2, activation='tanh', name='dec_dense5')
    dec_dense6 = Dense(cond_size2, activation='tanh', name='dec_dense6')
    dec_dense7 = Dense(cond_size2, activation='tanh', name='dec_dense7')
    dec_dense8 = Dense(cond_size2, activation='tanh', name='dec_dense8')
    #dec_dense6 = Bidirectional(CuDNNGRU(cond_size, return_sequences=True, name='dec_dense6'))
    #dec_dense7 = Bidirectional(CuDNNGRU(cond_size, return_sequences=True, name='dec_dense7'))
    #dec_dense8 = Bidirectional(CuDNNGRU(cond_size, return_sequences=True, name='dec_dense8'))

    dec_final = Dense(4*nb_used_features, activation='linear', name='dec_final')

    dec1 = dec_dense1(bits_input)
    dec2 = dec_dense2(dec1)
    dec3 = dec_dense3(dec2)
    dec4 = dec_dense4(dec3)
    dec5 = dec_dense5(dec4)
    dec6 = dec_dense6(dec5)
    dec7 = dec_dense7(dec6)
    dec8 = dec_dense8(dec7)
    output = Reshape((-1, nb_used_features))(dec_final(Concatenate()([dec1, dec2, dec3, dec4, dec5, dec6, dec7, dec8])))
    #output = dec_final(bits_input)
    #output = dec_final(noisy_bits)
    decoder = Model(bits_input, output, name='output')
    
    #hardquant = Activation(tf.keras.activations.hard_sigmoid)
    hardquant = Lambda(hard_quantize)
    e = encoder(feat)
    combined_output = decoder(hardquant(e))

    phony = Lambda(lambda x: 0*x[:,:,0:1])
    dist_params2 = Dense(2*nb_bits, activation='sigmoid', name='dist_dense2')
    dist2 = dist_params2(phony(e))
    e2 = Concatenate()([e, dist2])
    dist_params = Dense(2*nb_bits, activation='sigmoid', name='dist_dense')
    dist = dist_params(phony(e))
    e = Concatenate()([e, dist])

    model = Model(feat, [combined_output, e, e2])
    model.nb_used_features = nb_used_features

    return model, encoder, decoder

