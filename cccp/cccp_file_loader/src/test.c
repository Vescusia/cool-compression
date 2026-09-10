#include <time.h>
#include <unistd.h>

#include "lib.h"


int main(void) {
    FILE* sugoma = fopen("../blank-white.jpg", "rb");
    if (!sugoma) {
        printf("Error opening file\n");
        return 0;
    }

    FR_init(5, 2, sugoma);

    size_t sum = 0;

    const clock_t start = clock();

    for (int i = 0; i < 1000000000; i++) {
        const FR_batch_t batch = FR_get_batch();

        sum += batch.num_chunks;
        if (batch.num_chunks == 0) break;
    }

    const double secs = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("%lu in %f\n", sum, secs);
}
