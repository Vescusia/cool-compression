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
    # total_bytes = os.path.getsize(file_path)
    total_bytes = 0

    # two dims, list of alle relative distances between wrong predicted bits per chunk
    relative_indexes = []

    # initialize the file loader, set batch_size_bytes to a nice number :) (not important) BROOTTTTT
    loader = TorchFileLoader(file_path, chunk_size, batch_size_bytes)

    with torch.no_grad():
        # initialize model state
        state = model.init_state()

        # count false bits for evaluation, not important for usage (DEL)
        count_false_bits = 0

        # get data for evaluation (DEL?). not important for usage
        file_size = os.path.getsize(file_path)

        # progress bar for visual angucken
        bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)

        # counting processed batches, for early stoppen, if needed
        counter = 0

        while batch := loader.get_batch():
            inputs, targets = batch

            # update progress bar
            bar.update(len(inputs) * chunk_size)
            total_bytes += len(inputs) * chunk_size

            # predict next chunk
            predicted_chunks, state = model(inputs, state)
            predicted_chunks = torch.round(predicted_chunks)
            predicted_chunks = predicted_chunks.cpu().numpy()

            # do stuff
            targets = targets.cpu().numpy().ravel().astype(np.uint16)
            predicted_chunks = predicted_chunks.ravel().astype(np.uint16)

            # set all wrong bits as 1
            bool_array = (targets != predicted_chunks).astype(np.uint16)

            # count all false bits for evaluation (DEL)
            count_false_bits += np.sum(bool_array)

            # get array with only the indices of wrong bits
            index_array = np.argwhere(bool_array.ravel()).ravel().astype(np.uint64)

            # calc distances between adjacent wrong bits (relative indices of wrong bits)
            index_array = index_array - np.concat(([0], index_array[:-1]))
            index_array = index_array.astype(np.uint16)

            # subtract 1 from distances, as the bits have to be at least one apart
            index_array[1:] -= 1

            # add relative indices of wrong bits in this chunk to list for all chunks
            relative_indexes.append(index_array)

    bar.close()

    # Print Evaluation Results
    mean = np.mean([np.mean(batch_array) for batch_array in relative_indexes])
    print("mean: ", mean)
    print("std: ", np.mean([np.std(batch_array) for batch_array in relative_indexes]))
    print("max: ", np.max([np.max(batch_array) for batch_array in relative_indexes]))
    print(f"correct / false bits:    {total_bytes*8 - count_false_bits:,} / {count_false_bits:,}")
    print(f"required size:           {count_false_bits * (np.log2(round(mean)) + 1 + 1) / 8 + num_weights * 4:,.0f} B")
    print(f"file size in bytes/bits: {total_bytes:,} / {total_bytes * 8:,}")

    # plot relative indices
    plot(np.concat(relative_indexes))

    relative_indexes = np.array(relative_indexes[0], dtype=np.uint8)
    print(relative_indexes, relative_indexes.shape)
    enc_array = cccp_vle.npy_encoding(relative_indexes, 3)
    print(enc_array, enc_array.shape)

    np.save("data/hurricane.npy", enc_array)
    print("model saved")

    #dec_array = cccp_vle.npy_decoding(enc_array, 3)
    #print(dec_array, dec_array.shape)

    loaded_array = np.load("data/hurricane.npy", allow_pickle=True)
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
