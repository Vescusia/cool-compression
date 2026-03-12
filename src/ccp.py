# MMM BROT (Bio Brötchen????)
# BROTPAUSE??

import datetime
from pathlib import Path
from time import time

import click
import torch
import numpy as np
from tqdm import tqdm
import pytorch_optimizer

import model_manager
from model import LongMaster
import lib
from file_loader import BatchedDataLoader


BYTES_PER_STEP = 2 ** 15
CHUNKS_PER_BATCH = BYTES_PER_STEP // lib.CHUNK_SIZE
EPOCHS = 2000
OPTIMIZER_SWAP_EPOCHS = 500


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
LOGGER(f"Chunks per batch: {CHUNKS_PER_BATCH:,}")


@click.command()
@click.argument('input-path', type=click.Path(exists=True, dir_okay=False))
def main(input_path):
    # create model
    model = LongMaster()
    model.to(lib.DEVICE)

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_lstm_params = sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)
    num_res_net_params = sum(p.numel() for p in model.res_net.parameters() if p.requires_grad)
    LOGGER(f"Model parameters: {num_params:,} ({num_lstm_params:,} LSTM, {num_res_net_params:,} ResNet)")

    # define fast/first optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)

    # define loss function
    criterion = torch.nn.BCELoss()
    criterion.to(lib.DEVICE)

    # open the file to compress
    file_loader = BatchedDataLoader(input_path, CHUNKS_PER_BATCH)

    # train
    train_start = time()

    # create progressbar
    LOGGER(f"File size: {file_loader.file_size:,.0f} B")
    bar = tqdm(total=file_loader.file_size, unit='B', unit_scale=True)

    try:
        for epoch in range(EPOCHS):
            # reset epoch variables
            epoch_loss = 0.
            epoch_start = time()
            bar.reset()
            file_loader.reset()

            # (re-)initialize model state
            h, c = model.init_state()
            h, c = h.to(lib.DEVICE), c.to(lib.DEVICE)

            # keep track of time spent doing stuff
            total_batch_get_time = 0.
            total_train_time = 0.

            start_batch_get = time()
            while batch := file_loader.get_batch():
                # keep track of time spent doing stuff
                total_batch_get_time += time() - start_batch_get
                start_train = time()

                inputs, targets = batch

                bar.update(lib.CHUNK_SIZE * len(inputs))

                # predict next chunk
                predicted_chunks, h, c = model(inputs, h, c)
                h, c = h.detach(), c.detach()

                # calculate loss
                loss = criterion(predicted_chunks, targets)
                epoch_loss += loss.item()

                # backpropagate
                loss.backward()

                # step
                optim.step(lambda: loss)
                optim.zero_grad()

                # swap to slow optimizer
                if epoch == OPTIMIZER_SWAP_EPOCHS:
                    optim = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.)

                # keep track of time spent doing stuff
                total_train_time += time() - start_train
                start_batch_get = time()

            LOGGER(
                f"\nEpoch: {epoch}, "
                f"{'Fast' if epoch < OPTIMIZER_SWAP_EPOCHS else 'Slow'} optimizer, "
                f"Epoch loss: {epoch_loss:.2f}, "
                f"Epoch time: {time() - epoch_start:.2f} s, "
                f"Total time: {(time() - train_start) / 60:.2f} m"
                f"Batch get time: {total_batch_get_time / total_train_time:.2%}, "
                   )

            evaluate(input_path, model)

    except (Exception, KeyboardInterrupt) as e:
        LOGGER("Training stopped, saving model...")

        model_manager.save_model(model, Path('models'))
        model_manager.save_model_state_dict(model, Path('models'))

        raise e


def evaluate(input_path: str, model: torch.nn.Module):
    model.eval()

    file_loader = BatchedDataLoader(Path(input_path), CHUNKS_PER_BATCH)

    with torch.no_grad():
        total_correct = 0
        total_bits = 0

        # initialize model state
        h, c = model.init_state()
        h, c = h.to(lib.DEVICE), c.to(lib.DEVICE)

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
