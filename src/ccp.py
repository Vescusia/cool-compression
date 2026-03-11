# MMM BROT (Bio Sauerteigkruste)
# BROTPAUSE??

import datetime
import os
from pathlib import Path
from time import time

import click
import torch
import numpy as np
from tqdm import tqdm

from model import ByteMaster90, LongMaster

EPOCHS = 1000
CHUNK_SIZE = 128
CHUNKS_PER_BATCH = 4096
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FileLoader:
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, 'rb')
        self.file_size = os.path.getsize(path)
        self.chunks_per_batch = CHUNKS_PER_BATCH

        self.pos = None
        self.previous_chunk = None

    def __enter__(self):
        if self.file_size < CHUNK_SIZE:
            raise ValueError("File is smaller than chunk size")

        # reset the file and read the first chunk
        self.file.seek(0)
        self.pos = CHUNK_SIZE

        self.previous_chunk = self.file.read(CHUNK_SIZE)
        self.previous_chunk = self.norm_as_bytes(self.previous_chunk)

    def __exit__(self, *args, **kwargs):
        self.pos = CHUNK_SIZE

    def norm_as_bytes(self, chunk: bytes) -> torch.Tensor:
        # load chunk as floats
        chunk = np.frombuffer(chunk, dtype=np.uint8)
        chunk = chunk.astype(np.float32)
        # normalize to [0, 1]
        chunk /= 255
        # convert to tensor
        chunk = torch.from_numpy(chunk)

        # get index of every byte in the chunk
        indexes = np.arange(self.pos, self.pos + CHUNK_SIZE, dtype=np.float32)
        # normalize to [0, 1]
        indexes /= self.file_size
        # convert to tensor
        indexes = torch.from_numpy(indexes)

        # join tensors [a0, b0, a1, b1, ... ]
        normed = torch.stack((chunk, indexes), dim=1).flatten()
        return normed.to(DEVICE)

    @staticmethod
    def as_bits(raw_chunk: bytes) -> torch.Tensor:
        chunk = np.frombuffer(raw_chunk, dtype=np.uint8)

        chunk = np.unpackbits(chunk)
        chunk = chunk.astype(np.float32)

        chunk = torch.from_numpy(chunk)
        return chunk.to(DEVICE)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        # if we are at the end of the file, return None
        if self.pos >= self.file_size:
            return None

        inputs = []
        targets = []

        # collect all chunks
        for _ in range(self.chunks_per_batch):
            raw_chunk = self.file.read(CHUNK_SIZE)
            self.pos += CHUNK_SIZE

            # skip the last, (incomplete) chunk
            if len(raw_chunk) < CHUNK_SIZE:
                break

            target_chunk = self.as_bits(raw_chunk)
            targets.append(target_chunk)

            inputs.append(self.previous_chunk)
            self.previous_chunk = self.norm_as_bytes(raw_chunk)

        # convert to tensors
        inputs = torch.stack(inputs).to(DEVICE)
        targets = torch.stack(targets).to(DEVICE)

        return inputs, targets


class FilePrinter:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'a')

        print(f"Logging to {log_path}")
        print(f"\n{datetime.datetime.now().strftime('%H:%M %d.%m.%Y')}:\n", file=self.log_file, flush=True)

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, file=self.log_file, flush=True, **kwargs)


LOGGER = FilePrinter(Path('log.txt'))


@click.command()
@click.argument('input-path', type=click.Path(exists=True, dir_okay=False))
def main(input_path):
    # create model
    model = LongMaster()
    model.to(DEVICE)

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER(f"Model has {num_params:,} parameters")

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)

    # define loss function
    criterion = torch.nn.BCELoss()
    criterion.to(DEVICE)

    # open the file to compress
    file_loader = FileLoader(input_path)

    # train
    train_start = time()

    # create progressbar
    LOGGER(f"File size: {file_loader.file_size / 1024:,.0f} KiB")
    bar = tqdm(total=file_loader.file_size, unit='B', unit_scale=True)

    for epoch in range(EPOCHS):
        # reset epoch variables
        LOGGER(f"EPOCH: {epoch + 1}")
        epoch_loss = 0.
        epoch_start = time()
        bar.reset()

        # (re-)initialize model state
        h, c = model.init_state()
        h, c = h.to(DEVICE), c.to(DEVICE)

        with file_loader:
            while batch := file_loader.get_batch():
                inputs, targets = batch

                bar.update(CHUNK_SIZE * len(inputs))

                # predict next chunk
                predicted_chunks, h, c = model(inputs, h, c)
                h, c = h.detach(), c.detach()

                # calculate loss
                loss = criterion(predicted_chunks, targets)
                epoch_loss += loss.item()

                # backpropagate
                loss.backward()

                # optimize
                optimizer.step()
                optimizer.zero_grad()

        LOGGER(f"\nEpoch loss: {epoch_loss:.5f}, epoch time: {time() - epoch_start:.2f} s, total time: {(time() - train_start) / 60:.2f} m")
        evaluate(input_path, model)


def evaluate(input_path: str, model: torch.nn.Module):
    model.eval()

    file_loader = FileLoader(Path(input_path))

    with torch.no_grad():
        with file_loader:
            total_correct = 0
            total_bits = 0

            # initialize model state
            h, c = model.init_state()
            h, c = h.to(DEVICE), c.to(DEVICE)

            while batch := file_loader.get_batch():
                inputs, targets = batch

                # predict next chunks
                predicted_chunks, h, c = model(inputs, h, c)
                predicted_chunks = torch.round(predicted_chunks)
                predicted_chunks = predicted_chunks.cpu().numpy()

                # convert targets to numpy
                targets = targets.cpu().numpy()

                # calculate accuracy
                total_correct += np.sum(targets == predicted_chunks)
                total_bits += len(targets) * len(targets[0])

            LOGGER(f"{total_bits:,} Bits, {total_correct:,} correct, {total_correct / total_bits:.3%}")

    model.train()


if __name__ == '__main__':
    main()
