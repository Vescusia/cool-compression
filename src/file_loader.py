import os
from pathlib import Path
from torch.multiprocessing import spawn, Queue, ProcessContext

import numpy as np
import torch

import lib


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


def as_bits(raw_chunk: bytes) -> torch.Tensor:
    chunk = np.frombuffer(raw_chunk, dtype=np.uint8)

    chunk = np.unpackbits(chunk)
    chunk = chunk.astype(np.float32)

    chunk = torch.from_numpy(chunk)
    return chunk.to(lib.DEVICE)


def _next_packet_num(num: int) -> int:
    if num < 2 ** 20:
        return num + 1
    else:
        return 0


class ParallelLoader:
    def __init__(self, file_path: Path, chunk_size: int, batch_size_bytes: int, num_processors: int | None = None):
        self.batch_size = batch_size_bytes
        self.chunk_size = chunk_size
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)

        # Queues
        self.raw_data_q: Queue[tuple[int, bytes] | None] = Queue(maxsize=16)
        self.chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]] = Queue(maxsize=16)
        self.processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]] = Queue(maxsize=16)
        self.batch_q: Queue[tuple[torch.Tensor, torch.Tensor] | None] = Queue(maxsize=16)

        # start file reader
        self.file_reader: ProcessContext = spawn(self._load_file_thread, (file_path, self.raw_data_q, self.batch_size), 1, join=False, daemon=True)

        # start chunk orderer
        self.chunk_orderer: ProcessContext = spawn(self._order_chunks_thread, (self.raw_data_q, self.chunk_q, self.chunk_size), 1, join=False, daemon=True)

        # start processors
        self.num_processors = os.cpu_count() if num_processors is None else num_processors
        self.processors: ProcessContext = spawn(self._process_chunks_thread, (self.chunk_q, self.processed_chunk_q, self.file_size), self.num_processors, join=False, daemon=True)

        # start batchifier
        self.batchifier: ProcessContext = spawn(self._offset_processed_chunks_thread, (self.processed_chunk_q, self.batch_q), 1, join=False, daemon=True)

    @staticmethod
    def _load_file_thread(_, path: Path, raw_data_q: Queue[bytes | None], max_bytes_read: int):
        with open(path, 'rb') as file:
            while True:
                file.seek(0)

                while data := file.read(max_bytes_read):
                    raw_data_q.put(data, block=True)

                raw_data_q.put(None, block=True)

    @staticmethod
    def _order_chunks_thread(_, raw_data_q: Queue[bytes | None], chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]], chunk_size: int):
        overflow = np.array([], dtype=np.uint8)
        first_chunk_pos = 0  # should only count bytes from well-formed chunks

        chunk_num = 0

        while True:
            data = raw_data_q.get()

            # if the file has been read completely
            if data is None:
                # reset overflow and pos within file
                overflow = np.array([], dtype=np.uint8)
                first_chunk_pos = 0

                chunk_q.put((chunk_num, None), block=True)
                chunk_num = _next_packet_num(chunk_num)

                continue

            # convert to numpy
            data = np.frombuffer(data, dtype=np.uint8)

            # join last overflow
            data = np.concat((overflow, data), axis=0)

            # check that we have at least one chunk
            if len(data) < chunk_size:
                overflow = data
                continue

            # cut overflow
            overflow_size = len(data) % chunk_size
            if overflow_size > 0:
                overflow = data[-overflow_size:]
                data = data[:-overflow_size]
            else:
                overflow = np.array([], dtype=np.uint8)

            # split data into chunks
            num_chunks = len(data) // chunk_size
            assert len(data) / chunk_size == num_chunks
            chunks = np.split(data, num_chunks)
            chunks = np.array(chunks)

            # send to processors
            chunk_q.put((chunk_num, (first_chunk_pos, chunks)), block=True)
            chunk_num = _next_packet_num(chunk_num)

            # move position forward
            first_chunk_pos += len(data)

    @staticmethod
    def _process_chunks_thread(_process_index: int, chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]], processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], file_size: int):
        while True:
            num, data = chunk_q.get()

            if data is None:
                processed_chunk_q.put((num, None), block=True)
                continue

            # process to inputs
            first_chunk_pos, chunks = data
            inputs = ParallelLoader.chunks_to_inputs(chunks, first_chunk_pos, file_size)

            # process to targets
            targets = ParallelLoader.chunks_to_targets(chunks)

            # send to batchifier
            processed_chunk_q.put((num, (inputs, targets)), block=True)

    @staticmethod
    def _offset_processed_chunks_thread(_, processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], batch_q: Queue[tuple[torch.Tensor, torch.Tensor] | None]):
        tmp_packets = []
        current_packet_num = 0

        previous_last_input: torch.Tensor | None = None

        while True:
            # get new packet
            tmp_packets.append(processed_chunk_q.get())

            while True:
                # try to find the next packet
                for i, (num, data) in enumerate(tmp_packets):
                    if num == current_packet_num:
                        tmp_packets.pop(i)

                        # if the file ended
                        if data is None:
                            previous_last_input = None

                            batch_q.put(None, block=True)
                            current_packet_num = _next_packet_num(current_packet_num)

                            break

                        inputs, targets = data

                        # offset inputs and targets
                        if previous_last_input is None:
                            previous_last_input = torch.unsqueeze(inputs[-1], dim=0)

                            # cut off the first target and last input
                            inputs = inputs[:-1]
                            targets = targets[1:]

                        else:
                            # concat the previous last input to the front of inputs and cut off last input
                            next_last_input = torch.unsqueeze(inputs[-1], dim=0)
                            inputs = torch.concat((previous_last_input, inputs[:-1]))
                            previous_last_input = next_last_input

                        # send batch
                        batch_q.put((inputs, targets), block=True)
                        current_packet_num = _next_packet_num(current_packet_num)

                # append a new one
                else:
                    break

    def get_chunks(self):
        batch = self.batch_q.get()

        if batch is None:
            return None

        # move to GPU (cannot send GPU Tensors from thread to thread)
        inputs, targets = batch
        return inputs.to(lib.DEVICE), targets.to(lib.DEVICE)

    @staticmethod
    def chunks_to_inputs(chunks: np.ndarray, first_chunk_pos: int, file_size: int) -> torch.Tensor:
        # normalize to [0, 1]
        inputs = chunks.astype(np.float32)
        inputs /= 255

        # get index of every byte in the inputs
        end_index = first_chunk_pos + len(chunks) * len(chunks[0])  # emulating len(chunks) * chunk_size
        indexes = np.arange(first_chunk_pos, end_index, dtype=np.float32)

        # normalize to [0, 1]
        indexes /= file_size

        # reshape to chunks
        indexes = indexes.reshape(len(chunks), -1)

        # join tensors [a0, b0, a1, b1, ... ]
        inputs = np.stack((indexes, inputs), axis=2)
        inputs = inputs.reshape(len(chunks), -1, copy=False)

        # convert to tensor
        inputs = torch.from_numpy(inputs)

        return inputs

    @staticmethod
    def chunks_to_targets(chunks: np.ndarray) -> torch.Tensor:
        # turn targets to bits
        targets = np.unpackbits(chunks.ravel())

        # split to chunks
        targets = np.split(targets, len(chunks))
        targets = np.array(targets)

        # convert to tensor
        targets = targets.astype(np.float32)
        targets = torch.from_numpy(targets)

        return targets


if __name__ == "__main__":
    _file_path = Path("data") / "g2bb.jpg"

    _file_size = os.path.getsize(_file_path)
    print(f"File contains {_file_size:,} B")

    _chunk_size = 32
    loader = ParallelLoader(_file_path, _chunk_size, 2 ** 12 - 5, num_processors=1)

    # calculate the expected number of bytes to be predicted;
    # we only predict complete chunks, and the last/first one cannot be used as input/target
    expected_bytes = _chunk_size * (_file_size // _chunk_size - 1)

    while True:
        _total = 0

        while _data := loader.get_chunks():
            _inputs, _targets = _data

            _inputs = torch.flatten(_inputs)
            _total += len(_inputs) / 2

            print(torch.flatten(_targets).tolist())

        print(f"Expected total {expected_bytes:,} B")
        input(f"Seen total     {_total:,} B")
