# MMM BROT (Bio Brötchen????)
# BROTPAUSE??

import datetime
from pathlib import Path
from time import time

import click
import torch
import numpy as np
from torchviz import make_dot
from tqdm import tqdm

import model_manager
from model import LongMaster
import lib
from file_loader import ParallelLoader


BYTES_PER_STEP = 2 ** 20
EPOCHS = 200
OPTIMIZER_SWAP_EPOCHS = EPOCHS // 2
EVAL_EVERY_EPOCHS = 10
VISUALIZE_MODEL = False  # NEEDS GRAPHVIZ!!


class FilePrinter:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'a')

        print(f"\n{datetime.datetime.now().strftime('%H:%M %d.%m.%Y')}:\n", file=self.log_file, flush=True)

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)
        self.print_to_file(*args, **kwargs)

    def print_to_file(self, *args, **kwargs):
        print(*args, file=self.log_file, flush=True, **kwargs)


if __name__ == '__main__':
    LOGGER = FilePrinter(Path('log.txt'))
    LOGGER(f"Running on {lib.DEVICE}")


@click.command()
@click.argument('file-path', type=click.Path(exists=True, dir_okay=False))
def main(file_path):
    # create model
    model = LongMaster()
    model.compile()

    # initialize model weights
    model.apply(model.init_weights)
    model = model.to(lib.DEVICE)

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_lstm_params = sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)
    num_res_net_params = sum(p.numel() for p in model.res_net.parameters() if p.requires_grad)
    num_last_fc_params = sum(p.numel() for p in model.fc_to_output.parameters() if p.requires_grad)
    LOGGER(f"Model parameters: {num_params:,} ({num_lstm_params:,} LSTM, {num_res_net_params:,} ResNet, {num_last_fc_params:,} Last FC)")

    # define fast/first optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
    # optim = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=30)

    # define loss function
    criterion = torch.nn.BCELoss()
    criterion.to(lib.DEVICE)
    last_epoch_loss = 0.

    # open the file to compress
    file_loader = ParallelLoader(file_path, lib.CHUNK_SIZE, BYTES_PER_STEP)

    # create progressbars
    LOGGER(f"File size: {file_loader.file_size:,.0f} B")
    training_bar = tqdm(total=EPOCHS, unit=' Epochs', position=1)
    epoch_bar = tqdm(total=file_loader.file_size, unit=' B', unit_divisor=1024, unit_scale=True, position=0, leave=True)

    try:
        for epoch in range(EPOCHS):
            # reset epoch variables
            epoch_loss = 0.
            epoch_start = time()

            # update progressbars
            training_bar.update()
            epoch_bar.reset()

            # (re-)initialize model state
            state = model.init_state()

            # keep track of time spent doing stuff
            total_batch_get_time = 0.
            total_train_time = 0.
            start_batch_get = time()

            # train one the complete file once
            while batch := file_loader.get_chunks():
                # keep track of time spent doing stuff
                total_batch_get_time += time() - start_batch_get
                start_train = time()

                # unpack batch
                inputs, targets = batch
                epoch_bar.update(len(inputs) * lib.CHUNK_SIZE)

                # predict next chunk
                predicted_chunks, state = model(inputs, state)
                state = state.detach()

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
                    optim = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.)

                # keep track of time spent doing stuff
                total_train_time += time() - start_train
                start_batch_get = time()

            # epoch stats
            stats = f"{'Fast' if epoch < OPTIMIZER_SWAP_EPOCHS else 'Slow'} optimizer, " \
                    f"Epoch loss: {epoch_loss:.3f} ({epoch_loss - last_epoch_loss:.1e} delta), " \
                    f"Epoch time: {time() - epoch_start:.2f} s, " \
                    f"Batch get time: {total_batch_get_time / total_train_time:.2%}"
            last_epoch_loss = epoch_loss

            # display epoch stats
            epoch_bar.set_description_str(stats)
            LOGGER.print_to_file(stats)

            # evaluate regularly
            if epoch % EVAL_EVERY_EPOCHS == 0:
                evaluate(model, file_loader)

    finally:
        LOGGER("Training stopped, saving model...")

        # save state
        save_dir = Path('models')
        model_manager.save_model(model, save_dir)

        # save visualization
        if VISUALIZE_MODEL:
            dummy_input = torch.zeros((1, lib.CHUNK_SIZE * 2), device=lib.DEVICE)
            hx, cx = model.init_state()
            dummy_output = model(dummy_input, hx, cx)
            dot = make_dot(dummy_output, params=dict(model.named_parameters()))
            dot.format = 'png'
            dot.render(save_dir / f'viz_{model_manager.get_file_date()}')


def evaluate(model: torch.nn.Module, loader: ParallelLoader):
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_bits = 0

        # initialize model state
        state = model.init_state()

        # compute std deviation of predictions
        num_batches = 0
        pred_std = 0.
        pred_mean_diff = 0.
        last_pred_mean = None

        while batch := loader.get_chunks():
            inputs, targets = batch
            num_batches += 1

            # predict next chunks
            predicted_chunks, state = model(inputs, state)

            # compute prediction stats
            pred_std += predicted_chunks.std()
            if last_pred_mean is None:
                last_pred_mean = predicted_chunks.mean()
            else:
                pred_mean_diff += abs(predicted_chunks.mean() - last_pred_mean)

            # round
            predicted_chunks = torch.round(predicted_chunks)
            predicted_chunks = predicted_chunks.cpu().numpy()

            # convert targets to numpy
            targets = targets.cpu().numpy()

            # calculate accuracy
            total_correct += np.sum(targets == predicted_chunks)
            total_bits += len(targets) * len(targets[0])

        pred_std /= num_batches

        LOGGER(f"\n{total_bits:,} Bits, {total_correct:,} correct, {total_correct / total_bits:.3%}, (STD: {pred_std:.5}, DIFF: {pred_mean_diff:.5})")

    model.train()


if __name__ == '__main__':
    main()
