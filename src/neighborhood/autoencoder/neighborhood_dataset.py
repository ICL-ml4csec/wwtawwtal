"""Load neighborhoods as a time-series dataset for the LSTM VAE (or similar
models). This loads the entire dataset into memory, because they're small
enough that streaming is unnecessary."""

import logging

import torch

log = logging.getLogger(__name__)

class NbrDataset(torch.utils.data.Dataset):
    def __init__(self,
        mat: torch.tensor,
        window_size: int = 100,
    ):
        """Use `window_size`-long windows, where each window counts the number
        of neighborhoods in a bin. If window_size is None, automatically
        select the `window_size` returning a single sequence for the entire
        dataset.

        The input `mat` should have bins on the 0 axis, and neighborhoods on
        the 1 axis."""
        self.mat = mat
        if window_size is None:
            window_size = mat.shape[0]
        self.window_size = window_size
        self.n_windows = mat.shape[0] + 1 - window_size
        if self.n_windows <= 0:
            msg = (f"window size [{window_size}] " +
                   f"exceeds data length [{mat.shape[0]}]!")
            raise ValueError(msg)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, i):
        return self.mat[i:self.window_size + i]
