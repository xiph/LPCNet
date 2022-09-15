#ifndef _DUMP_DATA_H
#define _DUMP_DATA_H


#define MAX_GAIN_DB 0
#define MIN_GAIN_DB -30
/* #define APPY_RANDOM_GAIN_DROP */
#define APPLY_SILENCE


/* noise addition */
#define NO_NOISE 0
#define LEGACY_NOISE 1
#define VELVET_NOISE 2

#define NOISE_TYPE VELVET_NOISE

#if NOISE_TYPE == VELVET_NOISE
#define NUM_VELVET_NOISE 3

const int velvet_noise_T[NUM_VELVET_NOISE] = {5, 10, 20};
const int velvet_noise_a[NUM_VELVET_NOISE] = {1, 2, 3};
#endif

#endif