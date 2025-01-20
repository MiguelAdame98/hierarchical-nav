import h5py
import torch
from pathlib import Path

import numpy as np

from dommel_library.datasets.dataset import SequenceDataset, retry
from dommel_library.datastructs import TensorDict

import logging

logger = logging.getLogger(__name__)


class FilePool(SequenceDataset):
    """ FilePool stores sequences directly in h5 files in a directory
    Collects all h5 files in a given directory and uses these to retrieve data
    """

    def __init__(
        self,
        directory=None,
        transform=None,
        keys=None,
        num_workers=0,
        compression=9,
        interval=None,
        sequence_length=-1,
        sequence_stride=1,
        shuffle=False,
        cutoff=True,
        **kwargs,
    ):
        if not directory:
            directory = "/tmp"
        self._dir = Path(directory)
        self._keys = keys
        self._compression = compression
        if not self._dir.exists():
            self._dir.mkdir(parents=True)

        # list h5 files in directory
        self._files = list(sorted(self._dir.glob("*.h5")))

        # Train val test split
        if interval is None:
            interval = [0, 1.0]

        ind_min = int(interval[0] * len(self._files))
        ind_max = int(interval[1] * len(self._files))
        self._files = self._files[ind_min:ind_max]

        SequenceDataset.__init__(self, sequence_length, sequence_stride,
                                 shuffle, cutoff, transform, num_workers)

    def _load_sequences(self):
        lengths = [FilePool._get_length(file) for file in self._files]
        return self._files, lengths

    def _load_sequence(self, key, indices):
        return self._read(key, indices)

    @retry
    def _read(self, path, indices):
        sorted_indices = np.argsort(indices)
        inverse_sorted_indices = np.argsort(sorted_indices)

        with h5py.File(path, "r") as f:
            if not self._keys:
                self._keys = [k for k in f.keys() if len(f[k].shape) > 0]

            d = {}
            for key in self._keys:
                if self._sequence_length > 0:
                    # indices must be in ascending order for h5py
                    value = f[key][indices[sorted_indices]]
                    if self._shuffle:
                        # undo the sort needed for getting files
                        value = value[inverse_sorted_indices]
                else:
                    if key not in f.keys():
                        raise Exception("Key " + key + " not in " + str(path))
                    value = f[key][:]

                if value.dtype == "uint16":
                    value = value.astype(np.int32)
                d[key] = torch.as_tensor(value)
            return TensorDict(d)

    def push(self, sequence):
        for key, value in sequence.items():
            value.detach()
            # we squeeze the batch dimension here
            # so it can be added automatically when sampling
            if not self._keys or key in self._keys:
                sequence[key] = value.squeeze(0)

        # store to pickle
        path = self._dir / f"{len(self._files) + 1}.h5"
        # Avoid overwriting an existing file
        index = 2
        while path.exists():
            path = self._dir / f"{len(self._files) + index}.h5"
            index += 1
        self._push(path, sequence)
        self._files.append(path)
        self._update_table()

    @staticmethod
    @retry
    def _get_length(file):
        with h5py.File(file, "r") as fp:
            i = 0
            key = list(fp.keys())[i]
            while len(fp[key].shape) == 0:
                i += 1
                key = list(fp.keys())[i]
            return fp[key].shape[0]

    @retry
    def _push(self, file, sequence):
        with h5py.File(file, "w") as f:
            for key, value in sequence.items():
                f.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                    compression_opts=self._compression,
                )
