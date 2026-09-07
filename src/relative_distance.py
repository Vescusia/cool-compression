import numpy as np
import torch


class RelativeDistance:
    def __init__(self):
        self.previous_last_dist: int | None = None

    def reset(self):
        self.previous_last_dist = None

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

        # get indices of different bits
        different_bits = (predictions != targets).astype(np.uint16)
        different_bits = np.argwhere(different_bits).ravel()

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
        distances -= 1

        return distances
