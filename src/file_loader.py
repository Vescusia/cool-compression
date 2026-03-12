import os
import time
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import torch

import lib


class BatchedDataLoader:
    def __init__(self, path: Path, chunks_per_batch: int):
        self.path = path
        self.chunks_per_batch = chunks_per_batch

        # get file size
        self.file_size = os.path.getsize(path)
        if self.file_size < lib.CHUNK_SIZE:
            raise ValueError(f"File {path} is smaller than a chunk ({lib.CHUNK_SIZE} bytes)")

        # open file
        self.file = open(path, 'rb')
        self.previous_chunk = None
        self.pos = None
        self.reset()

    def reset(self):
        self.pos = 0
        self.file.seek(0)
        self.previous_chunk = self.norm_as_bytes(self.file.read(lib.CHUNK_SIZE), 0, self.file_size)

    @staticmethod
    def norm_as_bytes(chunk: bytes, chunk_start_pos: int, total_size: int) -> torch.Tensor:
        # load chunk as floats
        chunk = np.frombuffer(chunk, dtype=np.uint8)
        chunk = chunk.astype(np.float32)
        # normalize to [0, 1]
        chunk /= 255
        # convert to tensor
        chunk = torch.from_numpy(chunk)

        # get index of every byte in the chunk
        indexes = np.arange(chunk_start_pos, chunk_start_pos + lib.CHUNK_SIZE, dtype=np.float32)
        # normalize to [0, 1]
        indexes /= total_size
        # convert to tensor
        indexes = torch.from_numpy(indexes)

        # join tensors [a0, b0, a1, b1, ... ]
        normed = torch.stack((chunk, indexes), dim=1).flatten()
        return normed.to(lib.DEVICE)

    @staticmethod
    def as_bits(raw_chunk: bytes) -> torch.Tensor:
        chunk = np.frombuffer(raw_chunk, dtype=np.uint8)

        chunk = np.unpackbits(chunk)
        chunk = chunk.astype(np.float32)

        chunk = torch.from_numpy(chunk)
        return chunk.to(lib.DEVICE)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        # if we are at the end of the file, return None
        # pos is at the start of the last chunk, so if it is within the last chunk (most likely not fully lib.CHUNK_SIZE), we are at the end
        if self.pos >= self.file_size - lib.CHUNK_SIZE:
            return None

        inputs = []
        targets = []

        # collect all chunks
        for _ in range(self.chunks_per_batch):
            raw_chunk = self.file.read(lib.CHUNK_SIZE)

            # skip the last, (incomplete) chunk
            if len(raw_chunk) < lib.CHUNK_SIZE:
                break

            # process and append target (current) chunk
            target_chunk = self.as_bits(raw_chunk)
            targets.append(target_chunk)

            # append last (input) chunk
            inputs.append(self.previous_chunk)

            # process and current chunk as next last
            self.pos += lib.CHUNK_SIZE
            self.previous_chunk = self.norm_as_bytes(raw_chunk, self.pos, self.file_size)

        # convert to tensors
        inputs = torch.stack(inputs).to(lib.DEVICE)
        targets = torch.stack(targets).to(lib.DEVICE)

        return inputs, targets
