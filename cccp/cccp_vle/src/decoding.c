#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

struct decArrayTuple{
	size_t* array;
	int len;
};

struct decArrayTuple decompress(int bitlen, uint8_t* encArray, int storedBytes){

	// handle empty array -> return empty array, because no encoding can be done
	if (storedBytes == 0) {
		free(encArray);
		size_t* perArray = NULL;

		struct decArrayTuple tuple;
		tuple.array = perArray;
		tuple.len = storedBytes;

		return tuple;
	}

	int decArrayLen = storedBytes * 3;

	size_t* decArray = (size_t*) calloc(decArrayLen, sizeof(size_t));
	if (decArray == NULL) {
		printf("ERROR! memory allocation failed in decoding, decArray \n");
	}

	// get number of unused bits in last byte
	int unusedBits = encArray[0] >> 4;

	// counter for byte we are in. used as index in encoded Array
	int byteCount = 0;

	// read current byte to decode
	uint8_t curByte = encArray[byteCount];

	// lBits is number of already decoded Bits, rBits is number of Bits to decode in Byte.
	// first 4 bits are already used above, so skip these
	int lBits = 4;
	int rBits = 4;

	// count decoded relative indices
	int countRelIndices = 0;

	// start loop here?
	while(byteCount < storedBytes) {
		// exit loop if last sequence in last byte was decoded in
		if(byteCount == storedBytes -1 && rBits <= unusedBits){
			break;
		}

		// decoded relative index
		int decRelIndex = 0;
		uint8_t tmp = 0;

		// next sequence has to be joined to this sequence, at start "true" to enter loop 1 time
		bool join = true;

		// read in all further sequences to join them to relative index
		while(join){
			if (byteCount == storedBytes - 1 && rBits < unusedBits) {
				break;
			}

			if(rBits >= bitlen || rBits == 0){
				// if current Byte fully processed, read in next byte
				if(rBits == 0){
					lBits = 0;
					rBits = 8;
					curByte = encArray[++byteCount];
				}

				// cut off last bits, if they are from next sequence
				tmp = curByte >> (rBits - bitlen);
				// cut off all bits before this sequence
				tmp = tmp & (int) (pow(2, bitlen) - 1);

				// get prefix bit, which indicates if next sequence has to be joined or not
				if(tmp >> (bitlen - 1) == 1){
					join = true;
				} else{
					join = false;
				}

				// cut off prefix bit
				tmp = tmp & (int) (pow(2, bitlen - 1) - 1);

				// shift decoded relative index bitlen to left, to make space for sequence
				decRelIndex <<= (bitlen - 1);

				// join sequence to decoded relative index
				decRelIndex = decRelIndex | tmp;

				lBits = lBits + bitlen;
				rBits = rBits - bitlen;

			} else{
				// get sequence from byte
				tmp = curByte & (int) (pow(2, rBits) - 1);
				// get prefix bit to set join, for next sequence
				if(tmp >> (rBits - 1) == 1){
					join = true;
				} else{
					join = false;
				}
				// cut off all bits from prefix bit (included)
				tmp = tmp & (int) (pow(2, rBits - 1) - 1);
				// join sequence to decoded relative index
				decRelIndex <<= (rBits - 1);
				decRelIndex = decRelIndex | tmp;

				// get next byte
				uint8_t nextByte = encArray[++byteCount];

				// read rest of sequence from next byte
				tmp = nextByte >> (8 - (bitlen - rBits));
				// shift decoded relative index missing bits to left
				decRelIndex <<= (bitlen - rBits);
				// join rest of sequence to relative index
				decRelIndex |= tmp;

				// set current Byte to next Byte
				curByte = nextByte;
				lBits = bitlen - rBits;
				rBits = 8 - (bitlen - rBits);

			}
		}

		decArray[countRelIndices++] = decRelIndex;
		
	}

	// perfect array to return, so beautiful
	size_t* perArray = (size_t* ) malloc(sizeof(size_t) * countRelIndices);
	if (perArray == NULL) {
		printf("ERROR! memory allocation failed in decoding, perArray \n");
	}
	for(int i = 0; i < countRelIndices; i++){
		perArray[i] = decArray[i];
	}

	free(decArray); // checked

	struct decArrayTuple tuple;
	tuple.array = perArray;
	tuple.len = countRelIndices;

	return tuple;

}