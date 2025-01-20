import h5py
import os
import torch
import numpy as np
from collections import deque

from dommel_library.datasets.dataset import SequenceDataset, retry
from dommel_library.datastructs import TensorDict

import logging

logger = logging.getLogger(__name__)


class MemoryPool(SequenceDataset):
    """ MemoryPool stores sequences in a deque buffer in memory
    max_size specifies the maximum size of the buffer.
    The pool can be read / written from / to a directory of h5 files.
    """

    def __init__(
        self,
        max_size=None,
        transform=None,
        keys=None,
        num_workers=0,
        device="cpu",
        directory=None,
        interval=None,
        sequence_length=-1,
        sequence_stride=1,
        shuffle=False,
        cutoff=True,
        **kwargs,
    ):
        self._keys = keys
        self._buffer = deque(maxlen=max_size)
        self._device = torch.device(device)
        SequenceDataset.__init__(self, sequence_length, sequence_stride,
                                 shuffle, cutoff, transform, num_workers)
        if directory:
            self.load(directory, interval)

    def _load_sequences(self):
        keys = list(range(len(self._buffer)))
        lengths = [self._buffer[i].shape[0] for i in range(len(self._buffer))]
        return keys, lengths

    def _load_sequence(self, key, indices):
        raw_sequence = self._buffer[key]
        if indices.dtype.kind == "u":
            indices = indices.astype(np.int64)
        indices = torch.as_tensor(indices, dtype=torch.long)
        return raw_sequence[indices]

    def push(self, sequence):
        s = TensorDict({})
        for key, value in sequence.items():
            if self._keys is None or key in self._keys:
                value = value.detach()
                value = value.to(self._device)
                # we squeeze the batch dimension here (if present)
                # so it can be added automatically when sampling
                if value.shape[0] == 1:
                    value = value.squeeze(0)
                s[key] = value

        self._buffer.append(s)
        self._update_table()

    def dump(self, path, compression=9):
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(len(self._buffer)):
            file_name = str(i) + ".h5"
            out_path = os.path.join(path, file_name)
            sequence = self._buffer[i]
            with h5py.File(out_path, "w") as f:
                for key, value in sequence.items():
                    f.create_dataset(
                        key,
                        data=value,
                        compression="gzip",
                        compression_opts=compression,
                    )

    def load(self, path, interval=None):
        file_list = os.listdir(path)
        filtered_list = [item for item in file_list if ".h5" in item]

        # Train, validation split functionality
        if interval is None:
            interval = [0, 1.0]
        b = int(interval[0] * len(filtered_list))
        e = int(interval[1] * len(filtered_list))
        filtered_list = filtered_list[b:e]

        for name in filtered_list:
            file_path = os.path.join(path, name)
            d = self._load(file_path)
            self._buffer.append(TensorDict(d))

        self._update_table()
        return self

    @retry
    def _load(self, path):
        with h5py.File(path, "r") as f:
            # include all keys when no keys specified
            if not self._keys:
                keys = f.keys()
            else:
                keys = self._keys
            return {
                key: self._create_entry(value[:])
                for key, value in f.items()
                if key in keys
            }

    def _create_entry(self, value):
        value = value[:]
        if value.dtype == "uint16":
            value = value.astype(np.int32)
        return torch.as_tensor(value).to(self._device)

    def wrap(self, pool):
        for i in range(len(pool)):
            sequence = pool[i].unsqueeze(0)
            if self._keys is not None:
                filtered = {}
                for k, v in sequence.items():
                    if k in self._keys:
                        filtered[k] = v.detach().clone()
                sequence = TensorDict(filtered)
            self.push(sequence)
        return self
