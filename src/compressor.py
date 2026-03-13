import lib
from model import LongMaster
from model_manager import load_model, load_model_with_state_dict
from pathlib import Path
import torch
from file_loader import ParallelLoader
import file_loader
import os
import lib
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

    # set chunk size
    chunk_size = model.chunk_size
    batch_size_bytes = 2 ** 16

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
            bar.update(len(torch.flatten(inputs)) / 2)

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
            index_array = np.argwhere(bool_array.ravel()).ravel().astype(np.uint16)

            # calc distances between adjacent wrong bits (relative indices of wrong bits)
            index_array = index_array[1:] - index_array[:-1]

            # variable length encode relative indices

            # add relative indices of wrong bits in this chunk to list for all chunks
            relative_indexes.append(index_array)

            # increase batch counter
            counter += 1
            # if counter == 1000:
            #    break

        file.close()

    # print(relative_indexes)
    bar.close()

    #counter = 0
    mean = np.mean([np.mean(batch_array) for batch_array in relative_indexes])
    print("mean: ", mean)
    print("\nstd: ", np.std([np.std(chunk_array) for chunk_array in relative_indexes]))
    print("\nmax: ", np.max([np.max(chunk_array) for chunk_array in relative_indexes]))
    print("correct bits/false bits: ", counter * chunk_size * 8 - count_false_bits, "/", count_false_bits)
    print("bytes required:", count_false_bits * (np.log2(round(mean)) + 1) / 8)


if __name__ == "__main__":
    compress(Path("./data/log.txt"), Path("./models/model_202603122231.pt"))


