from pathlib import Path

import torch
import numpy as np
import tqdm
import click

from vle_loader import VLELoader
from model_manager import load_model
import lib


# CHANGE THIS TO WORK
FILE_SIZE: int | None = None


def predict_next_byte(model: torch.nn.Module, inputs: np.ndarray) -> np.uint8:
    # inputs to torch
    inputs = torch.from_numpy(inputs).to(lib.DEVICE)
    inputs = torch.unsqueeze(inputs, 0)

    # predict next byte
    pred_bits, state = model(inputs, predict_next_byte.state)
    predict_next_byte.state = state

    # round and convert to uint8
    pred_bits = torch.round(pred_bits[0])
    pred_bits = pred_bits.cpu().numpy().astype(np.uint8)

    # convert to byte
    pred_byte = np.packbits(pred_bits)
    return pred_byte[0].astype(np.uint8)


@click.command()
@click.argument('compressed-dir-path', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('decompressed-file-path', type=click.Path(dir_okay=False, file_okay=True))
def inflate(compressed_dir_path: str, decompressed_file_path: str):
    global FILE_SIZE

    # make paths ready
    decompressed_file_path = Path(decompressed_file_path)
    compressed_dir_path = Path(compressed_dir_path)

    # get file size and first chunk
    with open(compressed_dir_path / 'first_chunk.o', 'rb') as f:
        FILE_SIZE = int.from_bytes(f.read(8), byteorder='big', signed=False)
        print(f"File size: {FILE_SIZE / 1024:,.2f} kB")

        # read first chunk
        buf = bytearray()
        while len((new_bytes := f.read())) != 0:
            buf.extend(new_bytes)

        inputs = np.frombuffer(buf, dtype=np.float32)

    # load model
    model = load_model(compressed_dir_path / compressed_dir_path.with_suffix('').name)
    model = model.to(lib.DEVICE)
    predict_next_byte.state = model.init_state()

    # open file
    f = open(decompressed_file_path, 'wb')

    # write first chunk to file
    f.write((inputs[1:] * 255).astype(np.uint8).tobytes())

    # count predicted bytes
    num_bytes_written = len(inputs) - 1

    # create progress bar
    bar = tqdm.tqdm(total=FILE_SIZE, desc="Decompressing", unit="B", unit_scale=True, unit_divisor=1024)
    bar.update(num_bytes_written)

    with torch.no_grad():
        # initialize file loader to get relativ distances between wrong bits
        vle_fl = VLELoader(compressed_dir_path / 'relative_distances.npy')

        # get first predicted chunk (one byte)
        pred_byte = predict_next_byte(model, inputs)

        # behaves like the index of a false bit in the current byte. is used to flip the bit
        dist_to_wrong_bit = vle_fl.get_dist() - 1

        # loop over all relative distances
        while not vle_fl.is_finished:
            # correct pred byte
            while dist_to_wrong_bit < 8:
                # flip wrong bit
                pred_byte ^= 1 << (7 - dist_to_wrong_bit.astype(np.uint8))

                new_dist = vle_fl.get_dist()
                if new_dist is not None:
                    dist_to_wrong_bit += new_dist
                else:
                    break

            # save correct byte to file
            f.write(pred_byte.tobytes())
            num_bytes_written += 1
            bar.update(1)

            # get predicted chunk (one byte)
            inputs = np.concat(([(num_bytes_written + 1 - len(inputs)) / FILE_SIZE], inputs[2:], [pred_byte / 255]), dtype=np.float32)
            pred_byte = predict_next_byte(model, inputs)

            dist_to_wrong_bit -= 8

        # predict last correct bytes with model
        print(num_bytes_written)

        # write remaining, correctly predicted bytes to file
        for i in range(num_bytes_written, FILE_SIZE):
            inputs = np.concat(([(i + 1 - len(inputs)) / FILE_SIZE], inputs[2:], [pred_byte / 255]), dtype=np.float32)
            pred_byte = predict_next_byte(model, inputs)

            f.write(pred_byte.tobytes())

    f.close()


if __name__ == "__main__":
    inflate()
