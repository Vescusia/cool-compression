#ifndef CCPC_LIBRARY_H
#define CCPC_LIBRARY_H

#include <stdio.h>

#define BUF_SIZE (1 << 28)

extern size_t CHUNK_SIZE;
extern size_t CHUNKS_PER_BATCH;
#define CHUNK_SIZE_SHIFT 1

#define INPUT_CHUNK_SIZE (CHUNK_SIZE + 1)  // add index to start of chunk
#define INPUT_CHUNK_SIZE_BYTES (sizeof(float) * INPUT_CHUNK_SIZE)

#define TARGET_CHUNK_SIZE (CHUNK_SIZE_SHIFT * 8)
#define TARGET_CHUNK_SIZE_BYTES (sizeof(float) * TARGET_CHUNK_SIZE)

int fl_init(size_t,  size_t, FILE*);

typedef struct fl_batch_t {
    /// if num_chunks is 0, inputs and targets are NULL
    size_t num_chunks;
    /// has len of @code num_chunks * INPUT_CHUNK_SIZE_BYTES@endcode
    float* inputs;
    /// has len of @code num_chunks * TARGET_CHUNK_SIZE_BYTES@endcode
    float* targets;
} fl_batch_t;

fl_batch_t fl_get_batch(void);

#endif // CCPC_LIBRARY_H
