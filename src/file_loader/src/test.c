#include "lib.h"


int main(void) {
    FILE* sugoma = fopen("/home/vescusia/Programs/Python/cool-compression/data/tub_chem.bmp", "rb");
    if (!sugoma) {
        printf("Error opening file\n");
        return 0;
    }

    init(16, 24, sugoma);

    size_t sum = 0;

    for (int i = 0; i < 2; i++) {
        const batch_t batch = get_batch();
        sum += batch.num_chunks;
        if (batch.num_chunks == 0) break;
    }

    printf("%lu!\n", sum);
}
