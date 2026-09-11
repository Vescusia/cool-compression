#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "lib.h"


int main(void) {
    FILE* sugoma = fopen("../fish", "rb");
    if (!sugoma) {
        printf("Error opening file\n");
        return 0;
    }

    const size_t chunk_size = 5;
    const size_t chunks_per_batch = 2;

    // get file size
    fseek(sugoma, 0, SEEK_END);
    const size_t f_size = ftell(sugoma);
    fseek(sugoma, chunk_size, SEEK_SET);

    // read file into mem
    uint8_t* buf = malloc(f_size);
    while (fread(buf, 1, f_size, sugoma) != 0) {}
    fseek(sugoma, 0, SEEK_SET);

    // init file reader
    fl_init(chunk_size, chunks_per_batch, sugoma);

    size_t sum = 0;

    const clock_t start = clock();

    for (int i = 0; i < 1000000000; i++) {
        const fl_batch_t batch = fl_get_batch();

        // make sure that targets are the correct file bits
        for (size_t byte_i = 0; byte_i < batch.num_chunks; byte_i++) {
            uint8_t byte = 0;
            const uint8_t correct_byte = buf[sum + byte_i];

            for (uint8_t bit_i = 0; bit_i < 8; bit_i++) {
                byte += (uint8_t) batch.targets[byte_i * 8 + bit_i] << (7 - bit_i);
            }

            if (byte != correct_byte) {
                printf("Mismatch at %lu, %i - %i\n", sum + byte_i, byte, correct_byte);
            }
        }

        sum += batch.num_chunks;
        if (batch.num_chunks == 0) break;
    }

    const double secs = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("%lu in %f\n", sum, secs);

    // check that sum matches
    if (sum != f_size - chunk_size) {
        printf("Sum does not match\n");
    }

    free(buf);
}
