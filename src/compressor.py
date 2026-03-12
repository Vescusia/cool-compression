from model import LongMaster
from model_manager import load_model, load_model_with_state_dict
from pathlib import Path
import torch
from file_loader import BatchedDataLoader
from ccp import CHUNKS_PER_BATCH

import numpy as np

def compress(file_path: Path, model_path: Path = None):
    model = load_model(Path('models/model_202109221400.dict'))


    file_loader = BatchedDataLoader(file_path, CHUNKS_PER_BATCH)
    with torch.no_grad():
        # initialize model state
        h, c = model.init_state()

        with file_loader:

            while batch := file_loader.get_batch():
                inputs, targets = batch

                # predict next chunks
                predicted_chunks, h, c = model(inputs, h, c)
                predicted_chunks = torch.round(predicted_chunks)
                predicted_chunks = predicted_chunks.cpu().numpy()

                # convert targets to numpy
                targets = targets.cpu().numpy()

                # hier speicherung der falschen bits

                # get index of false bits
                bool_array = (targets != predicted_chunks)
                index_array = np.argwhere(bool_array).flatten()
                print(index_array)

                index_array = index_array[1:] - index_array[:-1]
                print(index_array)

                break


                
compress(Path("./data/fish"), Path("./models/model_202603121516.pt"))






