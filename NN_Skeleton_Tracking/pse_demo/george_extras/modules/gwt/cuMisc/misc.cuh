#define NUM_BLOCKS_MAX                      65535

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
#define NUM_RND_BURNIN                      1000

/*
 * Default grid/block sizes for the various functions.
 */

#define NUM_APPLY_BLOCKS                    4096
#define NUM_APPLY_THREADS_PER_BLOCK         512

#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define MUL24 __mul24
#define MIN(x, y) ((x) > (y) ? (y) : (x))

#define GTM_BLOCK_LOOPS_Y 32
#define GTM_LOOPY_BLOCK_LOOPS_Y 16

/*
 * The compiler is supposed to convert i / 16 to i >> 4, so
 * why is i >> 4 still so much faster?
 */
#define IDX(i) ((i) + ((i) >> 4))
//#define IDX(i) (i)

#define SSM_THREADS_X   16
#define SSM_THREADS_Y   32
#define SSM_LOOPS_Y     16

