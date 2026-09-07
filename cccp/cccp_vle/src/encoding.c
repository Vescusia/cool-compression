#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

struct encArrayTuple{
	uint8_t* array;
	int len;
};


struct encArrayTuple encoding(int bitlen, size_t* relIndexArr, int inputArrayLen){
	// encoding with bitlen bits per bit, -> bitlen-1 bits are saved in sequence

	// handle empty array -> return empty array, because no encoding can be done
	if (inputArrayLen == 0) {
		uint8_t* encRelIndexArr = NULL;
		free(relIndexArr);

		struct encArrayTuple tuple;
		tuple.array = encRelIndexArr;
		tuple.len = inputArrayLen;

		return tuple;
	}

	// raise error if bitlen invalid
	if(bitlen < 2 || bitlen > 8){
		printf("bitlen too small to store data (bitlen < 2) or too big for a byte (bitlen > 8) \n");
		exit(1);
	}

	// create and initialise bytearray to store encoded sequences of bits
	uint8_t* encArray = (uint8_t*) calloc(inputArrayLen * 2, sizeof(uint8_t));
	if (encArray == NULL) {
		printf("ERROR! memory allocation failed in encoding \n");
	}

	//counting added bytes
	int counter = 0;

	// byte in dem so viele sequenzen gespeichert werden, um später zu datei zu schreiben
	uint8_t encSeq = 0;

	// free bits from the left side in encSeq
	int lfreebits;

	// free bits from the right side in encSeq
	int rfreebits;

	// keep the first 4 bits to store the number of unused bits in last byte at the end
	if(bitlen > 4){
		lfreebits = 0;
		rfreebits = 4;
	} else{
		lfreebits = 4 - bitlen;
		rfreebits = bitlen;
	}

	for(int i = 0; i < inputArrayLen; i++){

		// get current relative index to encode
		size_t relindex = relIndexArr[i];
		
		// change zero to 1, to enter loop for the first time
		bool wasZero = false;
		if(relindex == 0){
			relindex = 1;
			wasZero = true;
		}

		// count bits that were cut off the end
		int cutOffBits = 0;
		// track if this is the first time in the while loop
		bool first = true;

		// if the relative index does not fit in bitlen-1 bits, create further sequences
		while(relindex != 0) {
			// check if relindex was zero to set it zero
			if(wasZero == true){
				relindex = 0;
			}

			// make a copy of relindex to work with
			uint8_t seqBits = relindex;
			// shift the index to right, so that only the first bitlen-1 bits remain at the right edge
			if (first){
				first = false;
				while (seqBits >= pow(2, (bitlen - 1))) {
					seqBits >>= (bitlen - 1);
					cutOffBits += (bitlen - 1);
				}
			} else{
				for (int t = 0; t < cutOffBits / (bitlen-1); t++) {
					seqBits >>= (bitlen - 1);
				}
			}

			// cut off the bits that will now be stored
			relindex &= (size_t) pow(2, cutOffBits) - 1;

			// set relindex to 1, to enter next loop, if trailing zeros are still to be stored
			if (relindex == 0 && cutOffBits > 0) {
				wasZero = true;
				relindex = 1;
			}

			// set prefix bit, if index was too big for this sequence
			if (cutOffBits > 0) {
				seqBits |= (int) pow(2, bitlen - 1);
			}
			cutOffBits -= (bitlen - 1);

			// add sequence of bits to encoded sequence (store in byte)
			if(rfreebits >= bitlen){
				encSeq = encSeq | seqBits;
				
				if(lfreebits >= bitlen){
					encSeq = encSeq << bitlen;
					lfreebits -= bitlen;
					rfreebits = bitlen;
				} else{
					encSeq = encSeq << lfreebits;
					rfreebits = lfreebits;
					lfreebits = 0;
				}
			} else{
				// initialise new encoded sequence
				uint8_t newEncSeq = 0;

				// get the bits that fit in the right free bits in old sequence
				int tmp = seqBits >> (bitlen - rfreebits);
				encSeq = encSeq | tmp;

				// cut off bits that are now stored in old sequence
				seqBits = seqBits & ((int) (pow(2, bitlen - rfreebits) - 1));
				
				// store übrige bits in new sequence
				newEncSeq = newEncSeq | seqBits;

				// shift im neuen byte bitlen weit nach links, wenn es passt
				// sonst shift nur so weit, sodass die sequenz am linken rand liegt
				if(8 - (bitlen - rfreebits) < bitlen){
					newEncSeq = newEncSeq << (8 - (bitlen - rfreebits));
					lfreebits = 0;
					rfreebits = 8 - (bitlen - rfreebits);
				} else{
					newEncSeq = newEncSeq << bitlen;
					lfreebits = 8 - bitlen - (bitlen - rfreebits);
					rfreebits = bitlen;
				}

				// store old sequence "encSeq" in array
				encArray[counter++] = encSeq;

				// set new byte
				encSeq = newEncSeq;


			}
		}
	}

	// shift bits in last byte to left edge
	encSeq = encSeq << lfreebits;

	// calc unused bits in last byte, from right side (max is 7)
	uint8_t unusedBits = lfreebits + rfreebits;
	if(unusedBits > 7){
		printf("ERROR! there are more unused bits in last byte than there should be. \n");
		exit(1);
	}

	// store unused Bits in last byte in the first 4 bits in the first byte. for decoding
	encArray[0] = encArray[0] | (unusedBits << 4);

	// add last byte to array
	encArray[counter++] = encSeq;

	// create array with perfect length to return
	uint8_t* encRelIndexArr = (uint8_t* ) malloc(counter * sizeof(uint8_t));
	for(int i = 0; i < counter; i++){
		encRelIndexArr[i] = encArray[i];
	}

	free(encArray); //checked

	struct encArrayTuple tuple;
	tuple.array = encRelIndexArr;
	tuple.len = counter;

	return tuple;

}