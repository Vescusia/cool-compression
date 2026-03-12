from torch.cuda import is_available


CHUNK_SIZE = 8  # in Bytes
DEVICE = 'cuda' if is_available() else 'cpu'
