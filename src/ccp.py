# MMM BROT (Bio Sauerteigkruste)
# Brotpause?

import os
from time import time

import click
import sklearn
import torch
import numpy as np
from tqdm import tqdm

from model import ByteMaster90, LongMaster


EPOCHS = 10
CHUNK_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def bitify(chunk: bytes) -> torch.Tensor:
    chunk = np.frombuffer(chunk, dtype=np.uint8)
    chunk = np.unpackbits(chunk)
    chunk = torch.from_numpy(chunk.astype(np.float32))
    return chunk.to(DEVICE)


def byteify(chunk: bytes) -> torch.Tensor:
    chunk = np.frombuffer(chunk, dtype=np.uint8)
    chunk = chunk.astype(np.float32)
    chunk /= 255
    chunk = torch.from_numpy(chunk)
    return chunk.to(DEVICE)


def indexify(total: int, chunk_start_pos: int, chunk: torch.Tensor) -> torch.Tensor:
    indexes = np.arange(chunk_start_pos, chunk_start_pos + CHUNK_SIZE, dtype=np.float32)
    indexes = indexes / total
    indexes = torch.stack((chunk, torch.from_numpy(indexes).to(DEVICE)), dim=1).flatten()
    return indexes


@click.command()
@click.argument('input-path', type=click.Path(exists=True, dir_okay=False))
def main(input_path):
    # create model
    model = LongMaster()
    model.to(DEVICE)

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters")

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)

    # define loss function
    criterion = torch.nn.BCELoss()
    criterion.to(DEVICE)

    # train
    train_start = time()

    with open(input_path, 'rb') as f:
        # get file size
        file_size = os.path.getsize(input_path)
        print(f"File has {file_size:,} B")

        # create progressbar
        bar = tqdm(total=file_size, unit='B', unit_scale=True)

        for epoch in range(EPOCHS):
            # reset epoch variables
            print(f"EPOCH: {epoch+1}")
            epoch_loss = 0.
            epoch_start = time()

            f.seek(0)
            total_read = 0
            bar.reset()

            # read first chunk
            previous_chunk = f.read(CHUNK_SIZE)
            previous_chunk = byteify(previous_chunk)

            # (re-)initialize model state
            h, c = model.init_state()

            while new_chunk := f.read(CHUNK_SIZE):
                bar.update(CHUNK_SIZE)
                total_read += CHUNK_SIZE

                if len(new_chunk) < CHUNK_SIZE:
                    break

                # bitify new chunk
                new_chunk = new_chunk

                # predict next chunk
                previous_chunk = indexify(file_size, total_read, previous_chunk)
                predicted_chunk, h, c = model(previous_chunk, h, c)
                h, c = h.detach(), c.detach()

                # calculate loss
                loss = criterion(predicted_chunk, bitify(new_chunk))
                epoch_loss += loss.item()

                # backpropagate
                loss.backward()

                # optimize
                optimizer.step()
                optimizer.zero_grad()

                # update previous chunk
                previous_chunk = byteify(new_chunk)

            print(f"Epoch loss: {epoch_loss:.5f}, epoch time: {time()-epoch_start:.2f} s, total time: {(time()-train_start) / 60:.2f} m")
            evaluate(input_path, model)


def evaluate(input_path: str, model: torch.nn.Module):
    model.eval()

    with open(input_path, 'rb') as f:
        file_size = os.path.getsize(input_path)
        total_read = 0

        with torch.no_grad():
            # read first chunk
            previous_chunk = f.read(CHUNK_SIZE)
            previous_chunk = byteify(previous_chunk)

            total_correct = 0
            total_bits = 0

            # initialize model state
            h, c = model.init_state()

            while new_chunk := f.read(CHUNK_SIZE):
                if len(new_chunk) < CHUNK_SIZE:
                    break
                total_read += CHUNK_SIZE

                # predict new chunk
                previous_chunk = indexify(file_size, total_read, previous_chunk)
                predicted_chunk, h, c = model(previous_chunk, h, c)
                predicted_chunk = torch.round(predicted_chunk)
                predicted_chunk = predicted_chunk.numpy()

                print(f"Plausible Std: {np.std(predicted_chunk)[0]:.3f}", end=', ')

                # get new chunk
                new_chunk = new_chunk

                # update previous chunk
                previous_chunk = byteify(new_chunk)

                # convert to numpy
                new_chunk = bitify(new_chunk).numpy()

                # calculate accuracy
                total_correct += np.sum(new_chunk == predicted_chunk)
                total_bits += len(new_chunk)

            print()
            print(f"{total_bits:,} Bits, {total_correct:,} correct, {total_correct / total_bits:.3%}")

    model.train()


if __name__ == '__main__':
    main()


