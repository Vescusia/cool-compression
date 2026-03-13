import os
from pathlib import Path
from multiprocessing import Process, Queue
from typing import Any

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
        self.file_reader = Process(target=self.new_load_file_thread,
                                   args=(file_path, self.raw_data_q, self.batch_size), daemon=True)
        self.file_reader.start()

        # start chunk orderer
        self.chunk_orderer = Process(target=self.order_chunks_thread, args=(self.raw_data_q, self.chunk_q, self.chunk_size), daemon=True)
        self.chunk_orderer.start()

        # start processors
        self.processors: list[Process] = []
        for _ in range(os.cpu_count() if num_processors is None else num_processors):
            p = Process(target=self.new_process_chunks_thread,
                        args=(self.chunk_q, self.processed_chunk_q, self.file_size), daemon=True)
            p.start()
            self.processors.append(p)

        # start batchifier
        self.batchifier = Process(target=self.offset_processed_chunks_thread, args=(self.processed_chunk_q, self.batch_q), daemon=True)
        self.batchifier.start()

    def __del__(self):
        # terminate processes on drop
        self.file_reader.terminate()
        self.chunk_orderer.terminate()

        for p in self.processors:
            p.terminate()

        self.batchifier.terminate()

    @staticmethod
    def next_packet_num(num: int) -> int:
        if num < 2 ** 20:
            return num + 1
        else:
            return 0

    @staticmethod
    def new_load_file_thread(path: Path, raw_data_q: Queue[bytes | None], max_bytes_read: int):
        with open(path, 'rb') as file:
            while True:
                file.seek(0)

                while data := file.read(max_bytes_read):
                    raw_data_q.put(data, block=True)

                raw_data_q.put(None, block=True)

    @staticmethod
    def order_chunks_thread(raw_data_q: Queue[bytes | None], chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]], chunk_size: int):
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
                chunk_num = ParallelLoader.next_packet_num(chunk_num)

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
            chunk_num = ParallelLoader.next_packet_num(chunk_num)

            # move position forward
            first_chunk_pos += len(data)

    @staticmethod
    def new_process_chunks_thread(chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]], processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], file_size: int):
        while True:
            num, data = chunk_q.get()

            if data is None:
                processed_chunk_q.put((num, None), block=True)
                continue

            # process to inputs
            first_chunk_pos, chunks = data
            inputs = ParallelLoader.process_to_inputs(chunks, first_chunk_pos, file_size)

            # process to targets
            targets = ParallelLoader.process_to_targets(chunks)

            # send to batchifier
            processed_chunk_q.put((num, (inputs, targets)), block=True)

    @staticmethod
    def offset_processed_chunks_thread(processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], batch_q: Queue[tuple[torch.Tensor, torch.Tensor] | None]):
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

                        # if file ended
                        if data is None:
                            previous_last_input = None

                            batch_q.put(None, block=True)
                            current_packet_num = ParallelLoader.next_packet_num(current_packet_num)

                            break

                        inputs, targets = data

                        # offset inputs and targets
                        if previous_last_input is None:
                            previous_last_input = torch.unsqueeze(inputs[-1], dim=0)

                            # cut off first target and last input
                            inputs = inputs[:-1]
                            targets = targets[1:]

                        else:
                            # concat previous last input to the front of inputs and cut off last input
                            next_last_input = torch.unsqueeze(inputs[-1], dim=0)
                            inputs = torch.concat((previous_last_input, inputs[:-1]))
                            previous_last_input = next_last_input

                        # send batch
                        batch_q.put((inputs, targets), block=True)
                        current_packet_num = ParallelLoader.next_packet_num(current_packet_num)

                # append a new one
                else:
                    break

    def new_get_chunks(self):
        return self.batch_q.get()

    @staticmethod
    def load_file_thread(path: Path, raw_q: Queue[tuple[int, bytes] | None], max_bytes_read: int):
        with open(path, 'rb') as file:
            while True:
                file.seek(0)
                pos = 0

                while data := file.read(max_bytes_read):
                    raw_q.put((pos, data), block=True)
                    pos += len(data)

                raw_q.put(None, block=True)

    @staticmethod
    def process_to_targets(chunks: np.ndarray) -> torch.Tensor:
        # turn targets to bits
        targets = np.unpackbits(chunks.ravel())

        # split to chunks
        targets = np.split(targets, len(chunks))
        targets = np.array(targets)

        # convert to tensor
        targets = targets.astype(np.float32)
        targets = torch.from_numpy(targets)

        return targets.to(lib.DEVICE)

    @staticmethod
    def process_to_inputs(chunks: np.ndarray, first_chunk_pos: int, file_size: int) -> torch.Tensor:
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

        return inputs.to(lib.DEVICE)

    @staticmethod
    def process_data_thread(raw_q: Queue[tuple[int, bytes] | None], processed_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], total_size: int, chunk_size: int):
        packet_num = 0
        overspill_data = np.array([], dtype=np.uint8)
        previous_last_input: torch.Tensor | None = None

        while True:
            # get raw data
            raw_bytes = raw_q.get()

            # handle file reset (completely read)
            if raw_bytes is None:
                processed_q.put((packet_num, None), block=True)
                packet_num = ParallelLoader.next_packet_num(packet_num)

                overspill_data = np.array([], dtype=np.uint8)
                previous_last_input = None

                continue

            # unpack data
            pos, raw_bytes = raw_bytes
            raw_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)

            # concatenate with overspill from last iteration
            raw_bytes = np.concat((overspill_data, raw_bytes))
            raw_bytes = np.array(raw_bytes)

            # if we have a previous last input, raw bytes needs to be at least one chunk such that we can
            # if we have no previous last input, raw byte needs to be at least two chunks for
            if (previous_last_input is not None and len(raw_bytes) < chunk_size) or (previous_last_input is None and len(raw_bytes) < chunk_size * 2):
                print("asdhashd")
                overspill_data = raw_bytes
                continue

            # check for overspill
            overspill = len(raw_bytes) % chunk_size
            if overspill != 0:
                overspill_data = raw_bytes[-overspill:]
                raw_bytes = raw_bytes[:-overspill]
            else:
                overspill_data = np.array([], dtype=np.uint8)

            # save overspill
            overspill_data = overspill_data

            # process targets
            targets = ParallelLoader.process_to_targets(raw_bytes, chunk_size)

            # process inputs
            inputs = ParallelLoader.process_to_inputs(raw_bytes, pos, total_size, chunk_size)

            # make sure that inputs and targets are properly offset
            if previous_last_input is not None:
                # concat previous last input to front to shift inputs over one chunk
                inputs = torch.concat((torch.unsqueeze(previous_last_input, 0), inputs[:-1]), dim=0)
                previous_last_input = inputs[-1]
            else:
                # cut of first target and last input
                previous_last_input = inputs[-1]
                inputs = inputs[:-1]
                targets = targets[1:]

            # send put on Queue
            processed_q.put((packet_num, (inputs, targets)), block=True)
            packet_num = ParallelLoader.next_packet_num(packet_num)

    def get_chunks(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        # try to find the next packet within the stored ones
        for i, (num, data) in enumerate(self.tmp_packets):
            if num == self.current_packet_num:
                self.tmp_packets.pop(i)
                self.inc_packet_num()

                return data

        # otherwise
        # get new packet
        self.tmp_packets.append(self.processed_chunk_q.get())

        # and try again
        return self.get_chunks()


if __name__ == "__main__":
    loader = ParallelLoader("data/g2bb.jpg", 256, 2 ** 12 - 5, num_processors=8)

    while True:
        total = 0

        while data := loader.new_get_chunks():
            inputs, targets = data

            inputs = torch.flatten(inputs)
            total += len(inputs) / 2

        input(total)
