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
import lib
import shutil

from relative_distance import RelativeDistance

@click.command()
@click.argument('compressed-dir-path', type=click.Path(exists=True))
@click.argument('decompressed-file-path', type=click.Path(exists=False, dir_okay=True), default='decompressed_data')
def inflate(compressed_dir_path: str, decompressed_file_path: str):
    # make paths ready
    decompressed_file_path = Path(decompressed_file_path)
    decompressed_file_path.mkdir(parents=True, exist_ok=True)
    compressed_dir_path = Path(compressed_dir_path)

    # get first chunk
    first_chunk = np.load(compressed_dir_path / 'first_chunk.npy')
    #print(first_chunk, first_chunk.shape)

    # load model
    model = load_model(compressed_dir_path / compressed_dir_path.with_suffix('').name)
    model = model.to(lib.DEVICE)

    with torch.no_grad():
        # initialize model state
        state = model.init_state()

        # get first input chunk
        inputs = torch.unsqueeze(torch.from_numpy(first_chunk), 0)

        # count predicted bytes
        pred_bytes = 0

        # load first file/batch of indices of wrong bits
        file_num = 0
        file_path = compressed_dir_path / "encoded" / f"{file_num:03}"
        loaded_array = np.load(file_path, allow_pickle=True)
        dec_array = cccp_vle.npy_decoding(loaded_array)
        dec_array += 1
        wrong_bit_indices = np.cumsum(dec_array, dtype=np.uint64)

        file_num += 1
        while (file_path := (compressed_dir_path / "encoded" / f"{file_num:03}")).exists():
            # load array and get indices for wrong predicted bit in batch
            loaded_array = np.load(file_path, allow_pickle=True)
            dec_array = cccp_vle.npy_decoding(loaded_array)
            dec_array += 1
            next_wrong_bit_indices = np.cumsum(dec_array, dtype=np.uint64)

            # get distance from last false bit in current batch to first false bit in next batch
            dist_to_next_chunk: int = next_wrong_bit_indices[0]

            # increase file counter (batch counter)
            file_num += 1


            while pred_bytes <= wrong_bit_indices[-1]:

                # get predicted chunk (one byte)
                predicted_bits, state = model(inputs, state)
                predicted_bits = torch.round(predicted_bits[0])




                pred_byte = np.packbits(predicted_byte.cpu().numpy().astype(np.uint8))
                #print(pred_byte)

                pred_byte += 1

                inputs = np.concat([pred_bits], inputs[2:], [pred_byte])

                print(inputs, inputs.shape)
                print(predicted_chunks, predicted_chunks.shape)


                exit()

if __name__ == "__main__":
    inflate()



