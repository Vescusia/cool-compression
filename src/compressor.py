import os
from pathlib import Path

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import click

from model_manager import load_model
from file_loader import TorchFileLoader
import cccp_vle
import shutil

from relative_distance import RelativeDistance


@click.command()
@click.argument('model-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('file-path', type=click.Path(exists=True, dir_okay=False))
def compress(model_path: str, file_path: str):
    # convert to Paths
    model_path = Path(model_path)
    file_path = Path(file_path)

    # load model
    if model_path.suffix == ".pt":
        model = load_model(Path(model_path))
        model = model.to('cpu')
    else:
        exit("fehler beim laden des models, weil vielleicht der pfad falsch ist!?")

    # get number of model weights
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # set chunk size
    chunk_size = model.chunk_size
    print(f"Chunk size: {chunk_size} B")
    batch_size_bytes = 2 ** 20

    # get file size
    file_size = os.path.getsize(file_path)

    # two dims, list of alle relative distances between wrong predicted bits per chunk
    relative_indexes = []

    # initialize the file loader, set batch_size_bytes to a nice number :) (not important) BROOTTTTT
    loader = TorchFileLoader(file_path, chunk_size, batch_size_bytes)

    # calculate relative distances
    rd = RelativeDistance()

    # store encoded distances
    encoded_distances = []

    with torch.no_grad():
        # initialize model state
        state = model.init_state()

        # progress bar for visual angucken
        bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)

        while batch := loader.get_batch():
            inputs, targets = batch

            # update progress bar
            bar.update(len(inputs) * chunk_size)

            # predict next chunk
            predicted_chunks, state = model(inputs, state)

            # do stuff
            distances = rd.to_relative(predicted_chunks, targets)
            relative_indexes.append(distances)

            # encode distances using variable length encoding
            encoded_distances.append(cccp_vle.npy_encoding(distances, 3))

    bar.close()

    # Print Evaluation Results
    mean = np.mean([np.mean(batch_array) for batch_array in relative_indexes])
    print("mean: ", mean)
    print("std: ", np.mean([np.std(batch_array) for batch_array in relative_indexes]))
    print("max: ", np.max([np.max(batch_array) for batch_array in relative_indexes]))
    print(f"correct / false bits:        {rd.total_bits - rd.total_incorrect_bits:,} / {rd.total_incorrect_bits:,}")
    print(f"est required size:           {rd.total_incorrect_bits * (np.log2(round(mean)) + 1 + 1) / 8 + num_weights * 4:,.0f} B")
    print(f"required size (vle):         {len(np.concat(encoded_distances)):,} B")
    print(f"required size (clementisch): {int(np.sum(np.concat(relative_indexes) + 1)) // 8:,} B")
    print(f"file size in B/b:            {file_size:,} / {rd.total_bits:,}")

    # plot relative indices
    plot(np.concat(relative_indexes))

    # build and create compression directory
    compression_path = model_path.with_suffix(model_path.suffix + '.ccp')
    compression_path.mkdir(parents=True, exist_ok=True)

    # save compressed data
    for i, encoded_array in enumerate(encoded_distances):
        np.save(compression_path / f'{i:03}.npy', encoded_array)
        shutil.copy(model_path, Path(compression_path) / model_path.name)


def inflate(compressed_dir_path: Path):
    for f in compressed_dir_path.iterdir():
        loaded_array = np.load(compressed_dir_path / f, allow_pickle=True)
        dec_array = cccp_vle.npy_decoding(loaded_array, 3)
        print(dec_array, dec_array.shape)


def plot(distances: np.ndarray):
    # plot the distribution of distances
    counts = np.bincount(distances.ravel())
    total_count = np.sum(counts)
    relative_frequencies = counts / total_count
    x_values = range(int(np.max(distances)) + 1)
    plt.bar(x_values, relative_frequencies)

    # add exponential curve with factor 1/2 in red
    x_curve = np.arange(0, int(np.max(distances)) + 1)
    y_curve = relative_frequencies[0] * (0.5 ** x_curve)
    plt.plot(x_curve, y_curve, 'r-', linewidth=1)

    plt.show()


if __name__ == "__main__":
    compress()
