import torch
import h5py as h5


def store_h5(data_dictionary, filename, compression='lzf'):
    """
    :param data_dictionary: dictionary that contains the data you want to store
    :param filename: Location to store data as h5 (<path/to/file.h5>)
    :param compression: type of compression to use (default='lzf')
    """
    with h5.File(filename, "w") as f:
        for key in data_dictionary.keys():
            f.create_dataset(
                key, data=data_dictionary[key], compression=compression
            )


def load_h5(filename):
    """
    :param filename: Location to load data from (<path/to/file.h5>)
    """
    data = {}
    with h5.File(filename, "r") as f:
        for key in f.keys():
            data[key] = torch.tensor(f[key][:], dtype=torch.float32)
    return data
