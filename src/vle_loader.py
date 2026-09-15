from pathlib import Path

import numpy as np

import cccp_vle


class VLELoader:
    def __init__(self, file_path: Path | str):
        # declare and define attributes
        self.dec_array = cccp_vle.npy_decoding(np.load(file_path, allow_pickle=True)) + 1
        self.bit_counter = 0

    @property
    def is_finished(self) -> bool:
        return self.bit_counter >= len(self.dec_array)

    def get_dist(self) -> np.uint16 | None:
        if self.is_finished:
            return None

        # get next number
        num = self.dec_array[self.bit_counter]
        self.bit_counter += 1
        return num
