from torch.cuda import is_available


CHUNK_SIZE = 32  # num of Bytes
CHUNK_SHIFT = 1
INPUT_CHUNK_SIZE = 33  # first is index within file
TARGET_CHUNK_SIZE = CHUNK_SHIFT * 8  # in bits

DEVICE = 'cuda' if is_available() else 'cpu'
