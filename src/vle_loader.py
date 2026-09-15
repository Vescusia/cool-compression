from pathlib import Path

import numpy as np

import cccp_vle


class VLELoader:
    def __init__(self, dir_path: Path | str):
        # declare attribute
        self.dec_array = None
        # path to dir with encoded batches
        self.dir_path = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        # index of current batch file
        self.count_batch = 0
        # prepare first batch
        self._load_next_batch()

    def _load_next_batch(self) -> bool:
        # make path to current batch file
        file_path = (self.dir_path / f"{self.count_batch:03}.npy")
        self.count_batch += 1
        # index of current bit to return in current batch
        self.bit_counter = 0

        # get next relative indices for batch
        if file_path.exists():
            loaded_array = np.load(file_path, allow_pickle=True)
            self.dec_array = cccp_vle.npy_decoding(loaded_array) + 1
            return True
        else:
            # HIER AUS Z.B. FIRST_CHUNK.NPY DIE FILE SIZE EINLESEN UND RÜKCGEBEN UM ENDE KORREKT ZU BEHANDELN
            self.dec_array = None
            return False

    def get_dist(self) -> np.uint16 | None:
        if self.dec_array is None:
            return None
        elif self.bit_counter < len(self.dec_array):
            num = self.dec_array[self.bit_counter]
            self.bit_counter += 1
            return num
        else:
            self._load_next_batch()
            return self.get_dist()

