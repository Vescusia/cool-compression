import os
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import torch

import lib


def normed_bytes_to_bits(normed_bytes: torch.Tensor) -> np.ndarray:
    """
    :param normed_bytes: 2d / 1d Tensor of normed ([0, 1] floats representing bytes)
    :return: flat array of the corresponding bits
    """
    # scale up to full bytes
    normed_bytes *= 255
    # round
    normed_bytes = torch.round(normed_bytes)
    # to numpy
    normed_bytes = normed_bytes.cpu().numpy()

    # to bits and flatten
    normed_bytes = normed_bytes.astype(np.uint8)
    normed_bytes = np.unpackbits(normed_bytes)

    return normed_bytes


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
        self.file_reader = Process(target=self._load_file_thread,
                                   args=(file_path, self.raw_data_q, self.batch_size), daemon=True)
        self.file_reader.start()

        # start chunk orderer
        self.chunk_orderer = Process(target=self._order_chunks_thread,
                                     args=(self.raw_data_q, self.chunk_q, self.chunk_size), daemon=True)
        self.chunk_orderer.start()

        # start processors
        self.processors: list[Process] = []
        for _ in range(os.cpu_count() if num_processors is None else num_processors):
            p = Process(target=self._process_chunks_thread,
                        args=(self.chunk_q, self.processed_chunk_q, self.file_size), daemon=True)
            p.start()
            self.processors.append(p)

        # start batchifier
        self.batchifier = Process(target=self._offset_processed_chunks_thread,
                                  args=(self.processed_chunk_q, self.batch_q), daemon=True)
        self.batchifier.start()

    def __del__(self):
        # terminate processes on drop
        self.file_reader.terminate()
        self.chunk_orderer.terminate()

        for p in self.processors:
            p.terminate()

        self.batchifier.terminate()

    @staticmethod
    def _load_file_thread(path: Path, raw_data_q: Queue[bytes | None], max_bytes_read: int):
        with open(path, 'rb') as file:
            while True:
                file.seek(0)

                while data := file.read(max_bytes_read):
                    raw_data_q.put(data, block=True)

                raw_data_q.put(None, block=True)

    @staticmethod
    def _order_chunks_thread(raw_data_q: Queue[bytes | None], chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]], chunk_size: int):
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
    def _process_chunks_thread(chunk_q: Queue[tuple[int, tuple[int, np.ndarray] | None]], processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], file_size: int):
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
    def _offset_processed_chunks_thread(processed_chunk_q: Queue[tuple[int, tuple[torch.Tensor, torch.Tensor] | None]], batch_q: Queue[tuple[torch.Tensor, torch.Tensor] | None]):
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
                            current_packet_num = _next_packet_num(current_packet_num)

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
                        current_packet_num = _next_packet_num(current_packet_num)

                # append a new one
                else:
                    break

    def get_chunks(self):
        return self.batch_q.get()

    @staticmethod
    def chunks_to_inputs(chunks: np.ndarray, first_chunk_pos: int, file_size: int) -> torch.Tensor:
        """
        :param chunks: 2d array of uint8's with the 2nd dimension being the chunk size
        :param first_chunk_pos: the position (within the file) of the first byte of the first chunk in ``chunks``
        :param file_size: the total file size in bytes
        :return: Tensor (on lib.DEVICE) of inputs
        """

        chunk_size = len(chunks[0])

        # turn targets to bits
        targets = np.unpackbits(chunks.ravel())

        # split to chunks
        targets = np.split(targets, len(chunks))
        targets = np.array(targets)

        # # create indexes
        # end_index = first_chunk_pos * 8 + len(chunks) * chunk_size * 8
        # indexes = np.arange(first_chunk_pos * 8, end_index, dtype=np.float32)
        #
        # # normalize and reshape indexes
        # indexes /= (file_size - chunk_size) * 8 - 1  # the last chunk is cut off
        # indexes = indexes.reshape(len(chunks), -1)
        #
        # # join indexes [a0, b0, a1, b1, ... ]
        # inputs = np.stack((indexes, targets), axis=2)
        # targets = inputs.reshape(len(chunks), -1, copy=False)

        # convert to tensor
        targets = targets.astype(np.float32)
        targets = torch.from_numpy(targets)

        return targets.to(lib.DEVICE)

    @staticmethod
    def chunks_to_targets(chunks: np.ndarray) -> torch.Tensor:
        # normalize to [0, 1]
        inputs = chunks.astype(np.float32)
        inputs /= 255

        # convert to tensor
        inputs = torch.from_numpy(inputs)

        return inputs.to(lib.DEVICE)


if __name__ == "__main__":
    loader = ParallelLoader(Path("data") / "g2bb.jpg", 16, 2 ** 5 - 3, num_processors=1)

    while True:
        _total = 0

        while _data := loader.get_chunks():
            _inputs, _targets = _data

            _targets = torch.flatten(_targets)
            _total += len(_targets)

            print(torch.flatten(_inputs).tolist())

        input(_total)
