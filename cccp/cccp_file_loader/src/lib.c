#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

#include "lib.h"


static FILE* file = NULL;
static size_t FILE_SIZE;

// file read buffer
static uint8_t buf[BUF_SIZE];
static size_t buf_data_start = 0;
static size_t buf_data_end = 0;
static size_t file_pos = 0;

size_t CHUNK_SIZE;
size_t CHUNKS_PER_BATCH;

// processed inputs
// dangerous restrict
// but should be fine, as there is no logical overlap because of the chunked data
static float* restrict inputs = NULL;
static float* previous_last_input_chunk = NULL;  // <--- references inputs

// processed targets
static float* restrict targets = NULL;


/**
 * @param chunk_size Size of the chunks in Bytes. Has to be >= 1. Has to be smaller than @code BUF_SIZE @endcode
 * @param chunks_per_batch Maximum chunks per batch. Returned batches can and will be smaller.
 * @param file_  File to be read from. Assumed to be seeked properly and not NULL.
 */
int fl_init(const size_t chunk_size, const size_t chunks_per_batch, FILE* file_) {
   CHUNK_SIZE = chunk_size;
   CHUNKS_PER_BATCH = chunks_per_batch;

   file = file_;

   // get file size
   fseek(file, 0, SEEK_END);
   FILE_SIZE = ftell(file);
   fseek(file, 0, SEEK_SET);

   // allocate input and target buffers
   inputs = calloc(INPUT_CHUNK_SIZE_BYTES, CHUNKS_PER_BATCH + 1);  // one more chunk for shifting inputs
   targets = calloc(TARGET_CHUNK_SIZE_BYTES, CHUNKS_PER_BATCH);

   if (inputs == NULL || targets == NULL) {
      errno = ENOMEM;
      return ENOMEM;
   }

   return 0;
}

/// The number of unread Bytes in buf
static size_t bytes_left(void) {
   return buf_data_end - buf_data_start;
}

fl_batch_t fl_get_batch(void) {
   // check if enough data for at least one chunk is in the buffer
   // this means it can potentially return nothing when previous last chunk is NULL (and we only get enough for one chunk)
   if (bytes_left() < CHUNK_SIZE) {
      // copy remaining bytes to front
      const size_t overflow = bytes_left();
      memmove(buf, buf + buf_data_start, overflow);
      buf_data_start = 0;

      // fill buffer
      const size_t read = fread(buf + overflow, 1, BUF_SIZE - overflow, file);
      buf_data_end = overflow + read;

      // check for EOF
      if (read == 0) {
         // reset state
         fseek(file, 0, SEEK_SET);
         previous_last_input_chunk = NULL;
         buf_data_end = 0;
         file_pos = 0;

         // return EOF batch
         return (fl_batch_t) {
            .num_chunks = 0,
            .inputs = NULL,
            .targets = NULL,
         };
      }

      // check that we have at least one chunk in the buffer
      if (buf_data_end - buf_data_start < CHUNK_SIZE) {
         return fl_get_batch();
      }
   }

   // calculate number of chunks for batch
   size_t num_chunks = (bytes_left() - CHUNK_SIZE) / CHUNK_SIZE_SHIFT + 1;
   if (num_chunks > CHUNKS_PER_BATCH) num_chunks = CHUNKS_PER_BATCH;

   // copy previous last input chunk to the front of this batch
   if (previous_last_input_chunk != NULL) {
      memmove(inputs, previous_last_input_chunk, INPUT_CHUNK_SIZE_BYTES);
   }

   // chunks to inputs
   for (size_t chunk = 0; chunk < num_chunks; chunk++) {
      const size_t input_chunk_start = (chunk + 1) * INPUT_CHUNK_SIZE;  // skip first input chunk

      // add the position within the file
      inputs[input_chunk_start] = (float) file_pos / (float) FILE_SIZE;
      file_pos += CHUNK_SIZE_SHIFT;

      // copy the bytes into inputs
      for (size_t byte_i = 0; byte_i < CHUNK_SIZE; byte_i++) {
         float transformed_byte = buf[buf_data_start + chunk * CHUNK_SIZE_SHIFT + byte_i];
         transformed_byte /= 255;

         // shift over by one chunk
         // such that the previous last chunk can be copied in
         inputs[input_chunk_start + byte_i + 1] = transformed_byte;
      }
   }

   // chunks to targets
   for (size_t chunk = 0; chunk < num_chunks; chunk++) {
      // copy bytes
      for (size_t byte_i = 0; byte_i < CHUNK_SIZE_SHIFT; byte_i++) {
         // get last CHUNK_SIZE_SHIFT bytes of this chunk
         const uint8_t raw_byte = buf[buf_data_start + chunk * CHUNK_SIZE_SHIFT + CHUNK_SIZE - CHUNK_SIZE_SHIFT + byte_i];

         // copy each bit
         // extract each bit from the byte
         uint8_t mask = 1 << 7;
         for (uint8_t bit_j = 0; bit_j < 8; bit_j++) {
            const uint8_t bit = (raw_byte & mask) > 0;

            targets[chunk * TARGET_CHUNK_SIZE + byte_i*8 + bit_j] = bit;

            mask >>= 1;
         }
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
   // we would usually have to shift by num_chunks-1 to properly reference the last chunk,
   // but with the placeholder initial chunk, this is equivalent.
   previous_last_input_chunk = inputs + num_chunks * INPUT_CHUNK_SIZE;
   // move buffer forward
   buf_data_start += num_chunks * CHUNK_SIZE_SHIFT;

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

   return (fl_batch_t) {
      .num_chunks = num_valid_chunks,
      .inputs = inputs_out,
      .targets = targets_out,
   };
}
