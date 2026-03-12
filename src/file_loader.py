import os
import time
from calendar import day_abbr
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

        self.max_pos = self.file_size - lib.CHUNK_SIZE - (self.file_size % lib.CHUNK_SIZE)

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
        if self.pos >= self.max_pos:
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


























def load_file_thread(path: Path, raw_q: Queue):
    messages_sent = 0
    with open(path, 'rb') as file:
        while True:
            file.seek(0)
            pos = 0

            while data := file.read(2 ** 15):
                raw_q.put((messages_sent, (pos, data)), block=True)
                messages_sent += 1
                pos += len(data)

            raw_q.put((messages_sent, None), block=True)
            messages_sent += 1


def process_data_thread(raw_q: Queue[tuple[int, tuple[int, bytes] | None]], processed_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], total_size: int):
    overspill_data = np.array([], dtype=np.uint8)

    while True:
        num, data = raw_q.get()
        if data is None:
            processed_q.put((num, None), block=True)
            continue

        # unpack data
        pos, data = data
        data = np.frombuffer(data, dtype=np.uint8)
        # concatenate the overspill from last iter
        data = np.concat((overspill_data, data))
        data = np.array(data)

        # check for overspill
        overspill = len(data) % lib.CHUNK_SIZE
        if overspill != 0:
            overspill_data = data[-overspill:]
            data = data[:-overspill]
        else:
            overspill_data = np.array([], dtype=np.uint8)

        # save overspill
        overspill_data = overspill_data

        # split data into chunks
#        print(f"Overspill: {overspill}, Ratio: {len(data) / lib.CHUNK_SIZE}")

        # offset inputs and targets
        # inputs = data[:-1]
        # targets = data[1:]

        # turn targets to bits
        targets = np.unpackbits(data)
        # split to chunks
        targets = np.split(targets, len(targets) // (lib.CHUNK_SIZE * 8))
        # skip first chunk
        targets = np.array(targets)[1:]
        # convert to tensor
        targets = targets.astype(np.float32)
        targets = torch.from_numpy(targets)
        targets = targets.to(lib.DEVICE)

        # normalize inputs
        inputs = np.split(data, len(data) // lib.CHUNK_SIZE)
        inputs = np.array(inputs)[1:]
        inputs = inputs.astype(np.float32)
        # normalize to [0, 1]
        inputs /= 255
        # convert to tensor
        inputs = torch.from_numpy(inputs)

        # get index of every byte in the inputs
        indexes = np.arange(pos, pos + lib.CHUNK_SIZE * len(inputs), dtype=np.float32)
        # normalize to [0, 1]
        indexes /= total_size
        # reshape to chunks
        indexes = indexes.reshape(-1, lib.CHUNK_SIZE)
        # convert to tensor
        indexes = torch.from_numpy(indexes)

        # join tensors [a0, b0, a1, b1, ... ]
        inputs = torch.stack((inputs, indexes), dim=1)
        inputs = inputs.flatten().to(lib.DEVICE)

        processed_q.put((num, (inputs, targets)), block=True)


if __name__ == '__main__':
    raw_q = Queue(maxsize=16)
    processed_q = Queue(maxsize=16)
    path = Path('data/tub_chem.bmp')

    file_size = os.path.getsize(Path('data/tub_chem.bmp'))
    print("size:", file_size)
    processer = Process(target=process_data_thread, args=(raw_q, processed_q, file_size), daemon=True)
    processer.start()

    for _ in range(7):
        p = (Process(target=process_data_thread, args=(raw_q, processed_q, file_size), daemon=True))
        p.start()

    time.sleep(2)

    loader = Process(target=load_file_thread, args=(path, raw_q), daemon=True)
    loader.start()

    current_num = 0
    tmp_packets = []
    last_batch = time.time()
    while True:
        ret = processed_q.get()

        tmp_packets.append(ret)

        for num, data in tmp_packets:
            if num == current_num:
                if data is None:
                    input("Done")
                else:
                    print(len(data[0]), len(tmp_packets))

                current_num += 1
                tmp_packets.remove((num, data))

                last_batch = time.time()
                break

        else:
            tmp_packets.append(ret)

