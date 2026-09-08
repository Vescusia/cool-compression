from torch.cuda import is_available


CHUNK_SIZE = 2  # in Bytes
DEVICE = 'cuda' if is_available() else 'cpu'
