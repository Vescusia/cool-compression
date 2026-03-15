from model import LongMaster
from model_manager import load_model, load_model_with_state_dict
from pathlib import Path
import torch
from file_loader import ParallelLoader
import os
import numpy as np
import tqdm


def compress(file_path: Path, model_path: Path = None):
    # load model
    if os.path.basename(model_path).split(".")[-1] == "pt":
        model = load_model(Path(model_path))
    elif os.path.basename(model_path).split(".")[-1] == "dict":
        model = load_model_with_state_dict(LongMaster, Path(model_path))
    else:
        exit("fehler beim laden des models, weil vielleicht der pfad falsch ist?")

    # get number of model weights
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # set chunk size
    chunk_size = model.chunk_size
    print(f"Chunk size: {chunk_size} B")
    batch_size_bytes = 2 ** 15

    # get file size
    # total_bytes = os.path.getsize(file_path)
    total_bytes = 0

    # two dims, list of alle relative distances between wrong predicted bits per chunk
    relative_indexes = []

    # initialise fileloader, set batch_size_bytes to a nice number :) (not important)
    file_loader = ParallelLoader(file_path, chunk_size, batch_size_bytes)

    with torch.no_grad():
        # initialize model state
        h, c = model.init_state()

        # count false bits for evaluation, not important for usage (DEL)
        count_false_bits = 0

        # get data for evaluation (DEL?). not important for usage
        file_size = os.path.getsize(file_path)

        # progress bar for visual angucken
        bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)

        # counting processed batches, for early stoppen, if needed
        counter = 0

        while batch := file_loader.get_chunks():
            inputs, targets = batch

            # update progress bar
            bar.update(len(inputs) * chunk_size)
            total_bytes += len(inputs) * chunk_size

            # predict next chunk
            predicted_chunks, h, c = model(inputs, h, c)
            predicted_chunks = torch.round(predicted_chunks)
            predicted_chunks = predicted_chunks.cpu().numpy()

            # do stuff
            targets = targets.cpu().numpy().ravel().astype(np.uint8)
            predicted_chunks = predicted_chunks.ravel().astype(np.uint8)

            # set all wrong bits as 1
            bool_array = (targets != predicted_chunks).astype(np.uint8)

            # count all false bits for evaluation (DEL)
            count_false_bits += np.sum(bool_array)

            # get array with only the indices of wrong bits
            index_array = np.argwhere(bool_array.ravel()).ravel().astype(np.uint64)

            # calc distances between adjacent wrong bits (relative indices of wrong bits)
            index_array = index_array - np.concat(([0], index_array[:-1]))

            # subtract 1 from distances, as the bits have to be at least one apart
            index_array[1:] -= 1

            # add relative indices of wrong bits in this chunk to list for all chunks
            relative_indexes.append(index_array)

    # print(relative_indexes)
    bar.close()

    # counter = 0
    mean = np.mean([np.mean(batch_array) for batch_array in relative_indexes])
    print("mean: ", mean)
    print("std: ", np.std([np.std(batch_array) for batch_array in relative_indexes]))
    print("max: ", np.max([np.max(batch_array) for batch_array in relative_indexes]))
    print(f"correct / false bits:    {total_bytes * 8 - count_false_bits:,} / {count_false_bits:,}")
    print(f"required size:           {count_false_bits * (np.log2(round(mean)) + 1) / 8 + num_weights * 4:,.0f} B")
    print(f"file size in bytes/bits: {total_bytes:,} / {total_bytes * 8:,}")


if __name__ == "__main__":
    compress(Path("./data/log.txt"), Path("models/model_2026_03.14_23-06.pt"))
