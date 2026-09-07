#include <time.h>
#include <unistd.h>

#include "lib.h"


int main(void) {
    FILE* sugoma = fopen("/home/vescusia/Programs/Python/cool-compression/data/tub_chem.bmp", "rb");
    if (!sugoma) {
        printf("Error opening file\n");
        return 0;
    }

    init(1, 1 << 15, sugoma);

    size_t sum = 0;

    const clock_t start = clock();

    for (int i = 0; i < 1000000000; i++) {
        const batch_t batch = get_batch();

        sum += batch.num_chunks;
        if (batch.num_chunks == 0) break;
    }

    const double secs = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("%lu in %f\n", sum, secs);
}
