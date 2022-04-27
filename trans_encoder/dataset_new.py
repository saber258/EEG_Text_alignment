import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, raw_data):

        self._signal = torch.FloatTensor(raw_data[:, 1:])
        self._label = torch.LongTensor(raw_data[:, 0])


    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._label)

    @property
    def sig_len(self):
        return self._signal.shape[1]

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._signal[idx], self._label[idx]
