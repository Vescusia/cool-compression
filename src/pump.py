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
@click.argument('compressed_dir', type=click.Path(exists=True))
def inflate(compressed_dir_path: Path):
    for f in compressed_dir_path.iterdir():
        loaded_array = np.load(compressed_dir_path / f, allow_pickle=True)
        dec_array = cccp_vle.npy_decoding(loaded_array, 3)
        print(dec_array, dec_array.shape)



