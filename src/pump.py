import os
from pathlib import Path

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import click

from file_loader.lib import vleFileLoader
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

    # GET FILE SIZE!!! in Bytes
    file_size = 209130

    # load model
    model = load_model(compressed_dir_path / compressed_dir_path.with_suffix('').name)
    model = model.to(lib.DEVICE)

    with open(decompressed_file_path / Path(compressed_dir_path.name).with_suffix('').with_suffix('').with_suffix(''), 'wb') as f:
        # write first chunk to file
        f.write((first_chunk[1:] * 255).astype(np.uint8).tobytes())
        # count predicted bytes
        count_bytes = len(first_chunk) - 1

        with torch.no_grad():
            # initialize model state
            state = model.init_state()

            # get first input chunk
            inputs = torch.unsqueeze(torch.from_numpy(first_chunk), 0)


            # initialize file loader to get relativ distances between wrong bits
            vle_fl = vleFileLoader(compressed_dir_path / "encoded")

            # get first predicted chunk (one byte)
            predicted_bits, state = model(inputs, state)
            predicted_bits = torch.round(predicted_bits[0])
            pred_byte = np.packbits(predicted_bits.cpu().numpy().astype(np.uint8)).astype(np.uint8)[0]

            # behaves like the index of a false bit in the current byte. is used to flip the bit
            bit_offset = vle_fl.get_dist() - 1

            # loop over all relative distances
            while vle_fl.dec_array is not None:

                while bit_offset < 8:
                    # flip wrong bit
                    pred_byte ^= 1 << (7 - bit_offset.astype(np.uint8))

                    new_dist = vle_fl.get_dist()
                    if new_dist is not None:
                        bit_offset += new_dist
                    else:
                        break

                # save correct byte to file
                f.write(pred_byte.tobytes())
                count_bytes += 1

                # get predicted chunk (one byte)
                inputs = np.concat(([(count_bytes + 1 - len(first_chunk)) / file_size], inputs[0][2:], [pred_byte / 255]), dtype=np.float32)
                inputs = np.reshape(inputs, (1, -1))
                predicted_bits, state = model(torch.from_numpy(inputs), state)
                predicted_bits = torch.round(predicted_bits[0])

                pred_byte = np.packbits(predicted_bits.cpu().numpy().astype(np.uint8)).astype(np.uint8)[0]

                bit_offset -= 8

            # predict last correct bytes with model
            print(count_bytes)

            for _ in range(count_bytes, file_size):

                predicted_bits, state = model(torch.from_numpy(inputs), state)
                predicted_bits = torch.round(predicted_bits[0])

                pred_byte = np.packbits(predicted_bits.cpu().numpy().astype(np.uint8)).astype(np.uint8)[0]

                f.write(pred_byte.tobytes())


if __name__ == "__main__":
    inflate()



