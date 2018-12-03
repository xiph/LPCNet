CFLAGS+=-Wall -W -O3 -g -W -Wextra

AVX2:=$(shell cat /proc/cpuinfo | grep -c avx2)
NEON:=$(shell cat /proc/cpuinfo | grep -c neon)

ifneq ($(AVX2),0)
CFLAGS+=-mavx2 -mfma
endif

ifneq ($(NEON),0)
CFLAGS+=-mfpu=neon
endif

all: dump_data test_lpcnet test_vec

dump_data: denoise.c kiss_fft.c pitch.c celt_lpc.c
	gcc -DTRAINING=1 $(CFLAGS) -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c -o dump_data -lm

nnet_data.o: nnet_data.c
	gcc $(CFLAGS) -c nnet_data.c

test_lpcnet: lpcnet.c nnet.c nnet_data.c nnet_data.h nnet_data.o
	gcc -o test_lpcnet $(CFLAGS) lpcnet.c nnet.c nnet_data.o -lm

test_vec: test_vec.c vec.h vec_neon.h vec_avx2.h
	gcc -o test_vec $(CFLAGS) test_vec.c -lm

test: test_vec
	./test_vec

clean:
	rm -f dump_data test_lpcnet test_vec
