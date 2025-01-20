import torch
from dommel_library.datastructs.dict import Dict
from dommel_library.distributions import MultivariateNormal
class TensorDict(Dict):
    """ Separate datastructure for mapping from str to torch.Tensor
        Allow to index both on key and slice
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            return dict.__getitem__(self, key)
        else:
            item = {}
            for k in self.keys():
                item[k] = dict.__getitem__(self, k)[key]
            return TensorDict(item)

    def to(self, device):
        for key, value in self.items():
            if value is not None:
                self[key] = value.to(device)
        return self

    def detach(self):
        for key, value in self.items():
            if value is not None:
                self[key] = value.detach()
        return self

    def squeeze(self, dim):
        """ Squeeze a dimension of the TensorDict """
        for key, value in self.items():
            self[key] = _check_type(value, value.squeeze(dim))
        return self

    def unsqueeze(self, dim):
        """ Unsqueeze a dimension of the TensorDict """
        for key, value in self.items():
            self[key] = _check_type(value, value.unsqueeze(dim))
        return self

    @property
    def shape(self):
        """ Get first two dimension shapes of the TensorDict values
        :return: first two dimensions, assuming these are batch and time
        """
        key = list(self.keys())[0]
        return tuple(self[key].shape[0: 2])


def cat(*dicts):
    """ Merge TensorDicts in the time dimension
    """
    if len(dicts) == 1:
        return dicts[0]
    merged = {}
    s1 = dicts[0]
    for i in range(1, len(dicts)):
        s2 = dicts[i]
        for key, value in s1.items():
            if key in s2:
                if value.shape[0] != s2[key].shape[0]:
                    # repeat in batch dimension if shapes are not the same
                    factor = int(s2[key].shape[0] / value.shape[0])
                    sizes = [factor]
                    for _ in range(len(value.shape) - 1):
                        sizes.append(1)
                    value = value.repeat(sizes)
                merged_value = torch.cat((value, s2[key]), dim=1)
                merged[key] = _check_type(value, merged_value)
            else:
                merged[key] = value
        s1 = merged

    return TensorDict(merged)


def stack(*dicts):
    """ Stack TensorDicts in the batch dimension """
    if len(dicts) == 1:
        return dicts[0]
    merged = {}
    s1 = dicts[0]
    for i in range(1, len(dicts)):
        s2 = dicts[i]
        for key, value in s1.items():
            if key in s2:
                merged_value = torch.cat((value, s2[key]), dim=0)
                merged[key] = _check_type(value, merged_value)
            else:
                merged[key] = value
        s1 = merged
    return TensorDict(merged)


def _check_type(value, result):
    """ Helper function to make sure distributions are also
    converted correctly """
    if value.__class__.__name__ != "Tensor":
        constructor = globals()[value.__class__.__name__]
        return constructor(result)
    return result
