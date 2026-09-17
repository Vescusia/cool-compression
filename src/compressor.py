import os
from pathlib import Path

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import click

import lib
from model_manager import load_model
from file_loader import TorchFileLoader
import cccp_vle
import shutil

from relative_distance import RelativeDistance


@click.command()
@click.argument('model-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('file-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--compress-to', type=click.Path(exists=False))
@click.option('--vle-bits', type=click.INT, default=2)
@click.option('--plot', 'show_plots', flag_value='plot', help='Display all incorrect bits of file')
def compress(model_path: str, file_path: str, compress_to: str, vle_bits: int, show_plots: bool):
    # convert to Paths
    model_path = Path(model_path)
    file_path = Path(file_path)

    # load model
    if model_path.suffix == ".pt":
        model = load_model(Path(model_path))
        model = model.to(lib.DEVICE)
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
    distances = []

    # initialize the file loader, set batch_size_bytes to a nice number :) (not important) BROOTTTTT
    loader = TorchFileLoader(file_path, chunk_size, batch_size_bytes, device=lib.DEVICE)

    # calculate relative distances
    rd = RelativeDistance()

    # store encoded distances
    encoded_distances = []

    # store first chunk to save on disk
    first_chunk = None

    with torch.no_grad():
        # progress bar for visual angucken
        bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)

        while batch := loader.get_batch():
            inputs, targets = batch

            # get first chunk
            if first_chunk is None:
                first_chunk = inputs[0].cpu().numpy().copy()

            # update progress bar
            bar.update(len(targets))

            # predict next chunk
            predicted_chunks = model(inputs)

            # do stuff
            new_distances = rd.to_relative(predicted_chunks, targets)
            distances.append(new_distances)

            # encode distances using variable length encoding
            encoded_distances.append(cccp_vle.npy_encoding(new_distances, vle_bits))

    bar.close()

    # many to one
    distances = np.concat(distances)

    # print evaluation results
    mean = float(np.mean(distances))
    bins = np.bincount(distances)

    # calculate entropy of distances
    total_entropy = calculate_entropy(bins)

    print("mean: ", mean)
    print("std: ", float(np.std(distances)))
    print("max: ", np.max([np.max(batch_array) for batch_array in distances]))
    print("uniques: ", int(np.sum(bins > 0)))
    print(f"correct / false bits:        {rd.total_correct_bits:,} / {rd.total_incorrect_bits:,} ({rd.total_correct_bits / rd.total_bits:.2%})")
    print(f"est required size:           {rd.total_incorrect_bits * (np.log2(round(mean)) + 1 + 1) / 8 + num_weights * 4:,.0f} B")
    print(f"required size (vle):         {len(np.concat(encoded_distances)):,} B")
    print(f"required size (clementisch): {int(np.sum(distances + 1)) // 8:,} B")
    print(f"file size in B/b:            {file_size:,} / {rd.total_bits:,}")
    print(f"compressed size:             {total_entropy / 8:,.0f} B ({total_entropy / rd.total_bits:.2%})")

    # plot relative indices
    if show_plots:
        print("Plotting file...")
        plot(distances, vle_bits)

    # build and create compression directory
    if compress_to is None:
        compression_dir = Path('compressed') / (file_path.stem + file_path.suffix + '.ccp')
    else:
        compression_dir = Path(compress_to)
    compression_dir.mkdir(parents=True, exist_ok=True)

    # save first chunk
    with open(compression_dir / 'first_chunk.o', 'wb') as f:
        f.write(file_size.to_bytes(8, byteorder='big', signed=False))
        f.write(first_chunk.tobytes())

    # copy model to compressed dir
    shutil.copy(model_path, compression_dir / 'model.pt')

    # save compressed data
    np.save(compression_dir / 'relative_distances.npy', cccp_vle.npy_encoding(distances, vle_bits))


def plot(distances: np.ndarray, vle_bits_per_bit: int):
    # plot the distribution of distances
    counts = np.bincount(distances.ravel())
    total_count = np.sum(counts)
    relative_frequencies = counts / total_count
    x_values = range(int(np.max(distances)) + 1)
    plt.bar(x_values, relative_frequencies)

    # calculate the total size (using vle) per distance
    sizes = []
    for i, count in enumerate(counts):
        numeric_bits = np.ceil(np.log2(i+1)) if i > 0 else 1
        bits_per_bit = np.ceil(numeric_bits / (vle_bits_per_bit-1)) * vle_bits_per_bit
        sizes.append(bits_per_bit * count)
    plt.bar(x_values, np.array(sizes) / np.sum(sizes), hatch='//', alpha=0.5)

    # add exponential curve with factor 1/2 in red
    x_curve = np.arange(0, int(np.max(distances)) + 1)
    y_curve = relative_frequencies[0] * (0.5 ** x_curve)
    plt.plot(x_curve, y_curve, 'r-', linewidth=1)

    plt.show()

    # create a second plot with the false bits as red points
    a = np.ceil(np.sqrt(np.sum(distances + 1)))  # sqrt of the length of all bits in the file (plot a square)
    distance_is = np.cumsum(distances.astype(np.uint32) + 1)
    x_points = distance_is % a
    y_points = distance_is // a

    plt.scatter(x_points, y_points, marker='s', color='tab:red', s=0.55)
    plt.show()


def calculate_entropy(bins: np.ndarray, cutoff: int = 9999) -> float:
    total_entropy: float = 0

    # cutoff any bin greater than cutoff
    bins = np.concat(([0], bins))
    for i, bin_ in enumerate(bins[cutoff+1:], start=cutoff+1):
        num_jumps, overflow = divmod(i, cutoff-1)

        bins[cutoff] += num_jumps * bin_
        bins[0] += num_jumps * bin_
        if overflow > 0:
            bins[0] += bin_
            bins[overflow] += bin_

        bins[i] = 0

    # calculate entropy
    n = np.sum(bins)
    for bin_ in filter(lambda x: x > 0, bins):
        bin_entropy = bin_ * -np.log2(bin_ / n)
        total_entropy += bin_entropy

    return total_entropy


if __name__ == "__main__":
    compress()
