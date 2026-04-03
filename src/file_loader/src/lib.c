#include "lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>


FILE* file = NULL;

uint8_t buf[BUF_SIZE];
size_t buf_pos = 0;
size_t buf_end = 0;

size_t CHUNK_SIZE;
size_t CHUNKS_PER_BATCH;

// dangerous restrict
// but should be fine, as there is no logical overlap because of the chunked data
float* restrict inputs = NULL;
float* previous_last_input_chunk = NULL;  // <--- references inputs

float* restrict targets = NULL;


/**
 * @param chunk_size Size of the chunks in Bytes. Has to be larger than 1. Has to be smaller than @code BUF_SIZE @endcode
 * @param chunks_per_batch Maximum chunks per batch. Returned batches can and will be smaller.
 * @param file_  File to be read from. Assumed to be seeked properly and not NULL.
 */
int init(const size_t chunk_size, const size_t chunks_per_batch, FILE* file_) {
   CHUNK_SIZE = chunk_size;
   CHUNKS_PER_BATCH = chunks_per_batch;

   file = file_;

   // allocate input and target buffers
   inputs = calloc(sizeof(float) * INPUT_CHUNK_SIZE, CHUNKS_PER_BATCH + 1);  // one more chunk for shifting inputs
   targets = calloc(sizeof(float) * TARGET_CHUNK_SIZE, CHUNKS_PER_BATCH);

   if (inputs == NULL || targets == NULL) {
      errno = ENOMEM;
      return ENOMEM;
   }

   return 0;
}


batch_t get_batch(void) {
   // check if enough data for at least one chunk are in the buffer
   // this means it can potentially return nothing when previous last chunk is NULL (and we only get enough for one chunk)
   if (buf_end - buf_pos < CHUNK_SIZE) {
      // copy remaining bytes to front
      const size_t overflow = buf_end - buf_pos;
      for (size_t i = buf_pos; i < buf_end; i++) {
         buf[i - buf_pos] = buf[i];
      }
      buf_pos = 0;

      // fill buffer
      const size_t read = fread(buf + overflow, 1, BUF_SIZE - overflow, file);
      buf_end = read + overflow;

      // check for EOF or error
      if (read == 0) {
         // reset state
         previous_last_input_chunk = NULL;
         fseek(file, 0, SEEK_SET);
         buf_pos = 0;
         buf_end = 0;

         // return EOF/error batch
         return (batch_t) {
            .num_chunks = 0,
            .inputs = NULL,
            .targets = NULL,
         };
      }

      // check that we have at least one chunk in the buffer
      if (buf_end - buf_pos < CHUNK_SIZE) {
         return get_batch();
      }
   }

   // calculate number of chunks for batch
   size_t num_chunks = (buf_end - buf_pos) / CHUNK_SIZE;
   if (num_chunks > CHUNKS_PER_BATCH) num_chunks = CHUNKS_PER_BATCH;
   const size_t num_batch_bytes = CHUNK_SIZE * num_chunks;

   // copy previous last input chunk to the front of this batch
   if (previous_last_input_chunk != NULL) {
      for (size_t i = 0; i < INPUT_CHUNK_SIZE; i++) {
         inputs[i] = previous_last_input_chunk[i];
      }
   }

   // chunks to inputs
   for (size_t i = 0; i < num_batch_bytes; i++) {
      float transformed_byte = buf[buf_pos + i];
      transformed_byte /= 255;

      // shift over by one chunk
      // such that the previous last chunk can be copied in
      inputs[i + INPUT_CHUNK_SIZE] = transformed_byte;
   }

   // chunks to targets
   for (size_t i = 0; i < num_batch_bytes; i++) {
      const uint8_t raw_byte = buf[buf_pos + i];

      // extract each bit from the byte
      uint8_t mask = 1 << 7;
      for (uint8_t j = 0; j < 8; j++) {
         const uint8_t bit = (raw_byte & mask) > 0;
         targets[i*8 + j] = (float)bit;

         mask >>= 1;
      }
   }

   float* inputs_out;
   float* targets_out;
   size_t num_valid_chunks;

   // offset input and target chunks
   if (previous_last_input_chunk == NULL) {
      // offset inputs without previous last chunk
      // reduced num_valid_chunks will cut off the last input chunk
      inputs_out = inputs + INPUT_CHUNK_SIZE;
      // cut off first targets chunk
      targets_out = targets + TARGET_CHUNK_SIZE;
      num_valid_chunks = num_chunks - 1;
   }
   else {
      // by default, the first input chunk is the previous last input chunk
      // that is already offset
      inputs_out = inputs;
      targets_out = targets;
      num_valid_chunks = num_chunks;
   }

   // store previous last input chunk
   // actually, we would want to shift by num_chunks - 1, to properly reference the last chunk,
   // but we have the placeholder initial chunk, so this is equivalent.
   previous_last_input_chunk = inputs + num_chunks * INPUT_CHUNK_SIZE;
   // move buffer forward
   buf_pos += num_chunks * CHUNK_SIZE;

   return (batch_t) {
      .num_chunks = num_valid_chunks,
      .inputs = inputs_out,
      .targets = targets_out,
   };

   // ---------------------------------------------------------------------------------------|
   // |                                   debug prints                                       |
   // ---------------------------------------------------------------------------------------|
   // printf("Inputs: [");
   // for (size_t i = 0; i < num_valid_chunks * INPUT_CHUNK_SIZE; i += INPUT_CHUNK_SIZE) {
   //    for (size_t j = 0; j < INPUT_CHUNK_SIZE; j++) {
   //       printf("%.3f", inputs_out[i+j]);
   //       printf(", ");
   //    }
   //    printf("| ");
   // }
   // printf("]\n");
   //
   // printf("Targets: [");
   // for (size_t i = 0; i < num_valid_chunks * TARGET_CHUNK_SIZE; i += TARGET_CHUNK_SIZE) {
   //    for (size_t j = 0; j < TARGET_CHUNK_SIZE; j += 8) {
   //       for (size_t k = 0; k < 8; k++) {
   //          printf("%.0f", targets_out[i+j+k]);
   //       }
   //       printf(", ");
   //    }
   //    printf("| ");
   // }
   // printf("]\n");
   //
   // return batch
}
