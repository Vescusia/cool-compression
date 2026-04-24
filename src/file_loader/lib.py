from pathlib import Path
from time import time, sleep

import numpy as np

# import c extension
import ccpc_file_loader


class FileLoader:
    def __init__(self, file_path: Path | str, chunk_size: int, batch_size_bytes: int):
        self.chunk_size = chunk_size
        self.chunks_per_batch = batch_size_bytes // chunk_size

        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)

        assert self.file_path.exists()
        assert self.file_path.is_file()
        self.file_size = self.file_path.stat().st_size

        ccpc_file_loader.init(self.chunk_size, self.chunks_per_batch, str(self.file_path.absolute()))

    @staticmethod
    def get_batch() -> tuple[np.ndarray, np.ndarray] | None:
        """
        :return: tuple of (inputs, targets) or `None` if the file has been fully read.
                 calling this function after it returned `None` will start reading from the beginning of the file again
        """
        return ccpc_file_loader.get_batch()


class TorchFileLoader:
    import torch

    def __init__(self, file_path: Path | str, chunk_size: int, batch_size_bytes: int, device: str = 'cpu', prefetch_bytes: int = 2 ** 30):
        self.chunk_size = chunk_size
        self.batch_size_bytes = batch_size_bytes

        self.device = self.torch.device(device)

        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.file_size = self.file_path.stat().st_size

        self.loader = FileLoader(file_path, chunk_size, batch_size_bytes)

        self.num_batch_batching = prefetch_bytes // batch_size_bytes
        self.is_first_prefetch = True
        self.is_completely_in_prefetch = False
        self.prefetched_batches: list[tuple[torch.Tensor, torch.Tensor] | None] = []

    def _get_batch_cpu(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        batch = self.loader.get_batch()

        if batch is None:
            return None

        # to Tensors
        inputs, targets = batch
        inputs = self.torch.from_numpy(inputs)
        targets = self.torch.from_numpy(targets)

        return inputs, targets

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        # just get a new batch for cpu
        if self.device.type == 'cpu':
            return self._get_batch_cpu()

        # --------------- GPU ---------------
        # take one of the prefetched batches
        if len(self.prefetched_batches) > 0:
            batch = self.prefetched_batches.pop(0)

            # if the prefetched batches contain the whole file,
            # just keep cycling within the prefetch
            if self.is_completely_in_prefetch:
                self.prefetched_batches.append(batch)

            return batch

        # prefetch multiple batches at once if using cuda
        for _ in range(self.num_batch_batching):
            batch = self._get_batch_cpu()

            # EOF
            if batch is None:
                self.prefetched_batches.append(None)
                break

            inputs, targets = batch

            # move to GPU
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.prefetched_batches.append((inputs, targets))

        # check if we prefetched the whole file
        if self.is_first_prefetch and self.prefetched_batches[-1] is None:
            print("File completely prefetched!")
            self.is_completely_in_prefetch = True
        self.is_first_prefetch = False

        return self.get_batch()


if __name__ == '__main__':
    # from file_loader_old import ParallelLoader
    import torch

    _path = Path('data/tub_chem.bmp')
    _total_size = _path.stat().st_size

    _loader = TorchFileLoader(_path, 8, 2 ** 20, device='cpu')
    # _loader_old = ParallelLoader(_path, 8, 2 ** 20, num_processors=7)

    _start = time()

    while _batch := _loader.get_batch():
        # _batch_old = _loader_old.get_chunks()
        # print(_batch[0].shape, _batch_old[0].shape)
        # assert torch.sum(_batch[0].flatten() != _batch_old[0].flatten()[1::2]) == 0
        # assert torch.sum(_batch[1] != _batch_old[1]) == 0
        pass

    _time_taken = time() - _start
    print(f"Time taken: {_time_taken:.2f} s ({_total_size / _time_taken / 1024 / 1024:,.5f} MiB/s)")
