# MMM BROT (Bio Brötchen????)
# BROTPAUSE?? JAAAAAAA

import datetime
from pathlib import Path
from time import time

import click
import torch
import numpy as np
from tqdm import tqdm

import cccp_vle
import model_manager
from model import Attanton63, ParamBuilder
import lib
from file_loader import TorchFileLoader
from relative_distance import RelativeDistance


EPOCHS = 100
EVAL_EVERY_EPOCHS = 5
ADAM_LR = .0001  # Attention needs wayyyy less LR
OPTIMIZER_SWAP_EPOCHS = EPOCHS // 2
SGD_LR = .0005
BYTES_PER_STEP = 2 ** 15
COMPILE = True


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
    # build model params
    params = (
        ParamBuilder().use_encoder(False)
        .use_decoder(False)
        .use_resnet(True)
        .with_embed_dim(36)
        .with_resnet_depth(4)
        .with_resnet_bottleneck(2)
        .with_heads(6)
        .build()
              )

    # create model
    model = Attanton63(params)
    if COMPILE:
        model.compile()

    # initialize model weights
    # model.apply(model.init_weights)
    model = model.to(lib.DEVICE)

    # print number of parameters
    model_manager.print_model_parameters(model)

    # define fast/first optimizer
    optim = torch.optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=0.)

    # define loss function
    criterion = torch.nn.BCELoss()
    criterion.to(lib.DEVICE)
    last_epoch_loss = 0.

    # open the file to compress
    file_loader = TorchFileLoader(file_path, lib.CHUNK_SIZE, BYTES_PER_STEP, device=lib.DEVICE)

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

            # keep track of time spent doing stuff
            total_batch_get_time = 0.
            total_train_time = 0.
            start_batch_get = time()

            # train on the complete file once
            while batch := file_loader.get_batch():
                # keep track of time spent doing stuff
                total_batch_get_time += time() - start_batch_get
                start_train = time()

                # unpack batch
                inputs, targets = batch
                epoch_bar.update(len(targets))

                # predict next chunk
                predicted_chunks = model(inputs)

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
                    optim = torch.optim.SGD(model.parameters(), lr=SGD_LR, weight_decay=0.)

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
        model_manager.save_model(model, save_dir, file_path)


def evaluate(model: torch.nn.Module, loader: TorchFileLoader):
    model.eval()

    # get number of model weights
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # compute relative distance between wrong predicted bits
    rd = RelativeDistance()

    # collect metrics
    mean_distances = []
    std_distances = []
    all_distances = []
    total_encoded_bytes = 0

    with torch.no_grad():
        # compute std deviation of predictions
        num_batches = 0
        pred_std = 0.
        pred_mean_diff = 0.
        last_pred_mean = None

        while batch := loader.get_batch():
            inputs, targets = batch
            num_batches += 1

            # predict next chunks
            predicted_chunks = model(inputs)

            # compute stochastic metrics
            pred_std += predicted_chunks.std()
            if last_pred_mean is None:
                last_pred_mean = predicted_chunks.mean()
            else:
                pred_mean_diff += abs(predicted_chunks.mean() - last_pred_mean)

            # calculate relative wrong bit distances
            if (distances := rd.to_relative(predicted_chunks, targets)) is not None:
                all_distances.append(distances)
                mean_distances.append(np.mean(distances))
                std_distances.append(np.std(distances))

                # encode to variable length encoded bits
                enc_array = cccp_vle.npy_encoding(distances, 2)
                total_encoded_bytes += len(enc_array)

        pred_std /= num_batches

        # estimate file size
        mean_distance = float(np.mean(mean_distances))
        std_distance = float(np.mean(std_distances))
        file_size = (np.log2(round(mean_distance)) + 1 + 1) * rd.total_incorrect_bits / 8 + num_params * 4

        LOGGER(
            f"\n{rd.total_bits:,} bits ({rd.total_bits // 8:,} B), "
            f"{rd.total_correct_bits:,} correct, "
            f"{rd.total_incorrect_bits:,} incorrect, "
            f"({rd.total_correct_bits / rd.total_bits:.3%}) "
            f"Mean Distance: {mean_distance:.2f}, Std Distance {std_distance:.2f}, "
            f"Pred Std: {pred_std:.5}, Batch Pred Diff: {pred_mean_diff:.5}, "
            f"Est File Size: {file_size:,.0f} B / {total_encoded_bytes:,.0f} B / {int(np.sum(np.concat(all_distances) + 1) / 8):,} B, "
            f"\n"
        )

    model.train()


if __name__ == '__main__':
    main()
