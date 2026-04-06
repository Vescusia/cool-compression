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
    import torch.multiprocessing as mp

    def __init__(self, file_path: Path | str, chunk_size: int, batch_size_bytes: int, device: str = 'cpu'):
        self.chunk_size = chunk_size
        self.batch_size_bytes = batch_size_bytes

        self.device = self.torch.device(device)

        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.file_size = self.file_path.stat().st_size

        self.loader = FileLoader(file_path, chunk_size, batch_size_bytes)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        batch = self.loader.get_batch()

        if batch is None:
            return None

        inputs, targets = batch

        # move to GPU
        inputs = self.torch.from_numpy(inputs).to(self.device, non_blocking=True)
        targets = self.torch.from_numpy(targets).to(self.device, non_blocking=True)

        return inputs, targets


if __name__ == '__main__':
    from file_loader_old import ParallelLoader
    # import torch

    _path = Path('data/tub_chem.bmp')
    _total_size = _path.stat().st_size

    _loader = TorchFileLoader(_path, 4, 2 ** 20, device='cpu')
    # _loader_old = ParallelLoader(_path, 4, 2 ** 20, num_processors=7)

    _start = time()

    while _batch := _loader.get_batch():
        # _batch_old = _loader_old.get_chunks()
        # print(_batch_old[0].shape)
        # assert torch.sum(_batch[1] != _batch_old[1]) == 0
        # assert torch.sum(_batch[0].flatten() != _batch_old[0].flatten()[1::2]) == 0
        pass

    _time_taken = time() - _start
    print(f"Time taken: {_time_taken:.2f} s ({_total_size / _time_taken / 1024 / 1024:,.5f} MiB/s)")
