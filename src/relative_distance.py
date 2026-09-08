import numpy as np
import torch


class RelativeDistance:
    def __init__(self):
        self.previous_last_dist: int | None = None
        self.total_bits: int = 0
        self.total_incorrect_bits: int = 0

    def reset(self):
        self.previous_last_dist = None
        self.total_bits = 0
        self.total_incorrect_bits = 0

    @property
    def total_correct_bits(self) -> int:
        return self.total_bits - self.total_incorrect_bits

    def to_relative(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """
        Compute the relative distance between the wrongly predicted bits.
        Call ``.reset()`` to reset the relative distance interchunk context.
        :param predictions: Tensor of unrounded predictions
        :param targets: Tensor of targets
        """

        # unpack predictions
        predictions = torch.round(predictions).cpu().numpy().ravel()

        # unpack targets
        targets = targets.cpu().numpy().ravel()
        self.total_bits += len(targets)

        # get indices of different bits
        different_bits = (predictions != targets)
        different_bits = np.argwhere(different_bits).ravel()
        self.total_incorrect_bits += len(different_bits)

        if len(different_bits) == 0:
            print("Perfect prediction")
            return np.array([])

        # calculate distances between different bits
        last_dist = len(targets) - different_bits[-1]
        if self.previous_last_dist is None:
            distances = different_bits - np.concatenate(([-1], different_bits[:-1]))
        else:
            distances = different_bits - np.concatenate(([-self.previous_last_dist], different_bits[:-1]))

        self.previous_last_dist = last_dist

        # reduce distance by one, always has to be at least one apart
        distances = distances.astype(np.uint8)
        distances -= 1

        return distances
