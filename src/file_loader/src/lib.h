#ifndef CCPC_LIBRARY_H
#define CCPC_LIBRARY_H

#include <stdio.h>

#define BUF_SIZE (1 << 28)

extern size_t CHUNK_SIZE;
extern size_t CHUNKS_PER_BATCH;

#define INPUT_CHUNK_SIZE (CHUNK_SIZE)
// one float for every bit
#define TARGET_CHUNK_SIZE (8 * CHUNK_SIZE)


int init(size_t,  size_t, FILE*);

typedef struct batch_t {
    /// if num chunks is 0, inputs and targets are NULL
    size_t num_chunks;
    /// has len of @code num_chunks * INPUT_CHUNK_SIZE @endcode
    float* inputs;
    /// has len of @code num_chunks * TARGET_CHUNK_SIZE @endcode
    float* targets;
} batch_t;

batch_t get_batch(void);

#endif // CCPC_LIBRARY_H
