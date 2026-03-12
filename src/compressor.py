import lib
from model import LongMaster
from model_manager import load_model, load_model_with_state_dict
from pathlib import Path
import torch
from file_loader import BatchedDataLoader
from ccp import CHUNKS_PER_BATCH
import os
import lib
import numpy as np
import tqdm


def compress(file_path: Path, model_path: Path = None):
    model = load_model(Path(model_path))
    chunk_size = lib.CHUNK_SIZE

    relative_indexes = []

    file_loader = BatchedDataLoader(file_path, CHUNKS_PER_BATCH)

    with torch.no_grad():
        # initialize model state
        h, c = model.init_state()

        count_false_bits = 0

        with open(file_path, "rb") as file:
            chunk = file.read(chunk_size)

            file_size = os.path.getsize(file_path)

            byte_counter = chunk_size

            bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)

            counter = 0
            while len(chunk) == chunk_size:
                inputs = file_loader.norm_as_bytes(chunk, byte_counter, file_size)
                byte_counter += chunk_size

                next_chunk = file.read(chunk_size)
                if len(next_chunk) != chunk_size:
                    break
                target_chunk = file_loader.as_bits(next_chunk)

                # predict next chunks
                inputs = torch.unsqueeze(inputs, 0)
                predicted_chunks, h, c = model(inputs, h, c)
                predicted_chunks = torch.round(predicted_chunks)
                predicted_chunks = predicted_chunks.cpu().numpy()

                # convert targets to numpy
                targets = target_chunk.cpu().numpy()

                # hier speicherung der falschen bits

                bar.update(chunk_size)
                # get indices of false bits
                bool_array = (targets != predicted_chunks).astype(np.uint8)
                count_false_bits += np.sum(bool_array)
                index_array = np.argwhere(bool_array.ravel()).ravel().astype(np.uint16)
                # print(index_array)

                # get relative indices
                index_array = index_array[1:] - index_array[:-1]

                relative_indexes.append(index_array)
                # print(relative_indexes)
                # print(index_array)

                counter += 1
                # if counter == 1000:
                #    break

        file.close()

    # print(relative_indexes)

    mean = np.mean([np.mean(chunk_array) for chunk_array in relative_indexes])
    print("mean: ", mean)
    print("\nstd: ", np.std([np.std(chunk_array) for chunk_array in relative_indexes]))
    print("\nmax: ", np.max([np.max(chunk_array) for chunk_array in relative_indexes]))
    print("correct bits/false bits: ", counter * chunk_size * 8 - count_false_bits, "/", count_false_bits)
    print("bytes required:", count_false_bits * (np.log2(round(mean)) + 1) / 8)



compress(Path("./data/ipsum.txt"), Path("./models/model_202603121935.pt"))


