# MMM BROT (Bio Sauerteigkruste)
import os
from time import time

import click
import sklearn
import torch
import numpy as np
from tqdm import tqdm

from model import ByteMaster90


EPOCHS = 10
CHUNK_SIZE = 1024


def bitify(chunk: bytes) -> torch.Tensor:
    chunk = np.frombuffer(chunk, dtype=np.uint8)
    chunk = np.unpackbits(chunk)
    chunk = torch.from_numpy(chunk.astype(np.float32))
    return chunk


@click.command()
@click.argument('input-path', type=click.Path(exists=True, dir_okay=False))
def main(input_path):
    # create model
    model = ByteMaster90()

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters")

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    # train
    train_start = time()

    with open(input_path, 'rb') as f:
        file_size = os.path.getsize(input_path)
        print(f"File has {file_size:,} B")

        for epoch in range(EPOCHS):
            print(f"EPOCH: {epoch+1}")
            epoch_loss = 0.
            epoch_start = time()
            f.seek(0)

            previous_chunk = f.read(CHUNK_SIZE)
            previous_chunk = bitify(previous_chunk)

            bar = tqdm(total=file_size, unit='B', unit_scale=True)

            while new_chunk := f.read(CHUNK_SIZE):
                bar.update(CHUNK_SIZE)
                if len(new_chunk) < CHUNK_SIZE:
                    break

                # bitify new chunk
                new_chunk = bitify(new_chunk)

                # predict next chunk
                predicted_chunk = model(previous_chunk)

                # calculate loss
                loss = criterion(predicted_chunk, new_chunk)
                epoch_loss += loss.item()

                # backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch loss: {epoch_loss:.5f}, epoch time: {time()-epoch_start:.2f} s, total time: {(time()-train_start) / 60:.2f} m")
            evaluate(input_path, model)


def evaluate(input_path: str, model: torch.nn.Module):
    model.eval()

    with open(input_path, 'rb') as f:
        with torch.no_grad():
            previous_chunk = f.read(CHUNK_SIZE)
            previous_chunk = bitify(previous_chunk)

            total_correct = 0
            total = 0

            while new_chunk := f.read(CHUNK_SIZE):
                if len(new_chunk) < CHUNK_SIZE:
                    break

                new_chunk = bitify(new_chunk)
                new_chunk = new_chunk.numpy()

                predicted_chunk = model(previous_chunk)
                predicted_chunk = np.round(predicted_chunk)
                predicted_chunk = predicted_chunk.numpy()

                total_correct += np.sum(new_chunk == predicted_chunk)
                total += len(new_chunk)

            print(f"{total} Bits, {total_correct} correct, {total_correct / total:.2%}")

    model.train()


if __name__ == '__main__':
    main()

