# LPCNet for FreeDV

Experimental version of LPCNet that has been used to develop FreeDV 2020 - a HF radio Digital Voice mode for over the air experimentation with Neural Net speech coding.  Possibly the first use of Neural Net speech coding in real world operation.

## Quickstart

```
$ cd ~
$ git clone https://github.com/drowe67/LPCNet.git
$ cd LPCNet && mkdir build_linux && cd build_linux
$ cmake ..
$ make
```

Unquantised LPCNet:

```
$ cd ~/LPCNet/build_linux/src
$ sox ../../wav/wia.wav -t raw -r 16000 - | ./dump_data --c2pitch --test - - | ./test_lpcnet - - | aplay -f S16_LE -r 16000
```

LPCNet at 1733 bits/s using direct-split quantiser:
```
sox ../../wav/wia.wav -t raw -r 16000 - | ./lpcnet_enc -s | ./lpcnet_dec -s | aplay -f S16_LE -r 16000
```

## Manually Selecting SIMD Technology

Cmake will select the fastest SIMD available (AVX/SSSE/None), however you can manually select e.g.:
```
make -DDISABLE_CPU_OPTIMIZATION=ON -DSSE=ON ..
```

## CTests

```
$ cd ~/LPCNet/build_linux
$ ctest
```

Note, due to precision/library issues several tests (1-3) will [only pass on some machines](https://github.com/drowe67/LPCNet/issues/17).

## Building Debian packages

To build Debian packages, simply run the "cpack" command after running "make". This will generate the following packages:

+ lpcnet: Contains the .so and .a files for linking/executing applications dependent on LPCNet.
* lpcnet-dev: Contains the header files for development using LPCNet.
* lpcnet-tools: Contains tools for use with LPCNet.

Once generated, they can be installed with "dpkg -i".

# Reading Further

1. [Original LPCNet Repo with more instructions and background](https://github.com/mozilla/LPCNet/)
1. [LPCNet: DSP-Boosted Neural Speech Synthesis](https://people.xiph.org/~jm/demo/lpcnet/)
1. [Sample model files](https://jmvalin.ca/misc_stuff/lpcnet_models/)

# Credits

Thanks [Jean-Marc Valin](https://people.xiph.org/~jm/demo/lpcnet/) for making LPCNet available, and [Richard](https://github.com/hobbes1069) for the CMake build system.

# Cross Compiling for Windows

This code has been cross compiled to Windows using Fedora Linux 30, see the freedv-gui README.md, and build_windows.sh script.

# Speech Material for Training

Suitable training material can be obtained from the McGill University Telecommunications & Signal Processing Laboratory. Download the ISO and extract the 16k-LP7 directory, the src/concat.sh script can be used to generate a headerless file of training samples.

```
cd 16k-LP7
sh /path/to/LPCNet/src/concat.sh
```

# Quantiser Experiments

The quantiser files used for these experiments (pred_v2.tgz and split.tgz) are [here](http://rowetel.com/downloads/deep/lpcnet_quant)

## Exploring Features

Install GNU Octave (if thats your thing).

Extract a feature file, fire up Octave, and mesh plot the 18 cepstrals for the first 100 frame (1 second):

```
$ ./dump_data --test speech_orig_16k.s16 speech_orig_16k_features.f32
$ cd src
$ octave --no-gui
octave:3> f=load_f32("../speech_orig_16k_features.f32",55);
nrows: 1080
octave:4> mesh(f(1:100,1:18))
```

## Uniform Quantisation

Listen to the effects of 4dB step uniform quantisation on cepstrals:

```
$ cat ~/Downloads/wia.wav | ./dump_data --test - - | ./quant_feat -u 4 | ./test_lpcnet - - | play -q -r 16000 -s -2 -t raw -
```

This lets us listen to the effect of quantisation error.  Once we think it sounds OK, we can compute the variance (average squared quantiser error). A 4dB step size means the error PDF is uniform in the range of -2 to +2 dB.  A uniform PDF has variance of (b-a)^2/12, so (2--2)^2/12 = 1.33 dB^2.  We can then try to design a quantiser (e.g. multi-stage VQ) to achieve that variance.

## Training a Predictive VQ

Clone and build [codec2](https://github.com/drowe67/codec2.git):

```
$ git clone https://github.com/drowe67/codec2.git
$ cd codec2 && mkdir build_linux && cd build_linux && cmake ../ && sudo make install
```

In train_pred2.sh, adjust PATH for the location of codec2-dev on your machine.

Generate 5E6 vectors using the -train option on dump_data to apply a bunch of different filters, then run the predictive VQ training script
```
$ cd LPCNet
$ ./dump_data --train all_speech.s16 all_speech_features_5e6.f32 /dev/null
$ ./train_pred2.sh
```

## Mbest VQ search

Keeps M best candidates after each stage:

```cat ~/Downloads/speech_orig_16k.s16 | ./dump_data --test - - | ./quant_feat --mbest 5 -q pred2_stage1.f32,pred2_stage2.f32,pred2_stage3.f32 > /dev/null```

In this example, the VQ error variance was reduced from 2.68 to 2.28 dB^2 (I think equivalent to 3 bits), and the number of outliers >2dB reduced from 15% to 10%.

## Streaming of WIA broadcast material

Interesting mix of speakers and recording conditions, some not so great microphones. Faster speech than the training material.

Basic unquantised LPCNet model:

```sox -r 16000 ~/Downloads/wianews-2019-01-20.s16 -t raw - trim 200 | ./dump_data --c2pitch --test - - | ./test_lpcnet - - | aplay -f S16_LE -r 16000```

Fully quantised at (44+8)/0.03 = 1733 bits/s:

```sox -r 16000 ~/Downloads/wianews-2019-01-20.s16 -t raw - trim 200 | ./dump_data --c2pitch --test - - | ./quant_feat -g 0.25 -o 6 -d 3 -w --mbest 5 -q pred_v2_stage1.f32,pred_v2_stage2.f32,pred_v2_stage3.f32,pred_v2_stage4.f32 | ./test_lpcnet - - | aplay -f S16_LE -r 16000```

## Fully quantised encoder/decoder programs

Same thing as above with quantisation code packaged up into library functions.  Between quant_enc and quant_dec are 52 bit frames every 30ms:

```cat ~/Downloads/speech_orig_16k.s16 | ./dump_data --c2pitch --test - - | ./quant_enc | ./quant_dec | ./test_lpcnet - - | aplay -f S16_LE -r 16000```

Same thing with everything integrated into stand alone encoder and decoder programs:

```cat ~/Downloads/speech_orig_16k.s16 | ./lpcnet_enc | ./lpcnet_dec | aplay -f S16_LE -r 16000```

The bit stream interface is 1 bit/char, as I find that convenient for my digital voice over radio experiments.  The decimation rate, number of VQ stages, and a few other parameters can be set as command line options, for example 20ms frame rate, 3 stage VQ (2050 bits/s):

```cat ~/Downloads/speech_orig_16k.s16 | ./lpcnet_enc -d 2 -n 3 | ./lpcnet_dec -d 2 -n 3 | aplay -f S16_LE -r 16000```

You'll need the same set of parameters for the encoder as decoder.

Useful additions would be:

1. Run time loading of .h5 NN models.
1. A --packed option to pack the quantised bits tightly, which would make the programs useful for storage applications.

## Direct Split VQ

Four stage VQ of log magnitudes (Ly), 11 bits (2048 entries) per stage, First 3 stages 18 elements wide; final stage 12 elements wide.  During training this acheived similar variance to 4 stage predictive quantiser (measured on 12 bands).  Same bit rate, but direct quantisation means more robust to bit errors and especially packet loss.

```
sox ~/Desktop/deep/quant/wia.wav -t raw - | ./dump_data --c2pitch --test - - | ./quant_feat -d 3 -i -p 0 --mbest 5 -q split_stage1.f32,split_stage2.f32,split_stage3.f32,split_stage4.f32 | ./test_lpcnet - - | aplay -f S16_LE -r 16000
```

Compare this to four stage predictive VQ of Cepstrals (DCT of Ly), 11 bits (2048 entries) per stage, 18 element wide vectors.  We quantise the predictor output.

```
sox ~/Desktop/deep/quant/wia.wav -t raw -  | ./dump_data --c2pitch --test - - | ./quant_feat -d 3 -w --mbest 5 -q pred_v2_stage1.f32,pred_v2_stage2.f32,pred_v2_stage3.f32,pred_v2_stage4.f32 | ./test_lpcnet - - | aplay -f S16_LE -r 16000
```

Both are decimated by a factor of 3 (so 30ms update of parameters, 30*44=1733 bits/s).

# Effect of Bit Errors

Random 1 Bit Error Rate (BER):

Predictive:
```sox wav/wia.wav -t raw -r 16000 - | ./lpcnet_enc | ./lpcnet_dec -b 0.01 | aplay -f S16_LE -r 16000```

Direct-split:
```sox wav/wia.wav -t raw -r 16000 - | ./lpcnet_enc -s | ./lpcnet_dec -s -b 0.01 | aplay -f S16_LE -r 16000```
