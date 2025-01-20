import torch
from torch.nn.functional import interpolate

from dommel_library.datastructs import TensorDict


class RandomObservation:
    """Select random single observations from a sequence."""

    def __call__(self, sequence):
        observations = sequence["observation"]
        i = torch.randint(observations.shape[0], (1,))
        return observations[i, :].squeeze(0), i


class RandomSubsequence:
    """Select random subsequence from a sequence."""

    def __init__(self, length):
        self.length = length

    def __call__(self, sequence):
        subsequence = {}

        # assume all sequences have an action item to get the size
        actions = sequence["action"]
        size = actions.shape[0]

        if size - self.length <= 0:
            print(
                "Warning: sequence is not long enough"
                "to sample a subsequence..."
            )
            return sequence

        i = int(torch.randint(size - self.length, (1,)).item())
        for key, value in sequence.items():
            # unsqueeze a dimension in case we have a vector
            if len(value.shape) == 1:
                value = value.unsqueeze(-1)
            subsequence[key] = value[i: i + self.length, ...]

        return TensorDict(subsequence)


class Subsample:
    """ Subsample from sequence """

    def __init__(self, subsample_step):
        self.subsample_step = subsample_step

    def __call__(self, sequence):
        subsequence = {}
        for key, value in sequence.items():
            if len(value.shape) == 1:
                value = value.unsqueeze(-1)
            subsequence[key] = value[:: self.subsample_step, ...]
        return TensorDict(subsequence)


class Crop:
    """ Crop images in a sequence"""

    def __init__(self, start_point, end_point, keys):
        self.startx = start_point[0]
        self.starty = start_point[1]
        self.endx = end_point[0]
        self.endy = end_point[1]
        self.keys = keys

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                cropped_sequence = sequence[key][
                    :,  # noqa: W503,W504
                    :,  # noqa: W503,W504
                    self.startx: self.endx,  # noqa: W503,W504
                    self.starty: self.endy,  # noqa: W503,W504
                    ...,  # noqa: W503,W504
                ]
                sequence[key] = cropped_sequence
        return sequence


class ChannelFirst:
    """ switch tensors from channel last to channel first """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                # first dimension is sequence
                switched_sequence = sequence[key].permute(0, 3, 1, 2)
                sequence[key] = switched_sequence
        return sequence


class ToFloat:
    def __init__(self, keys=None):
        self.keys = keys
        pass

    def __call__(self, sequence):
        if not self.keys:
            keys = sequence.keys()
        else:
            keys = self.keys

        for key in keys:
            if key in sequence.keys():
                sequence[key] = sequence[key].float()
        return sequence


class Pad:
    """ adds padding to the data, changes the shape"""

    def __init__(self, pad_amount, keys):
        self.pad_amount = pad_amount
        self.keys = keys
        self.padder = torch.nn.ReplicationPad2d(pad_amount)

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                sequence[key] = self.padder(sequence[key]).detach()
        return sequence


class RescaleShift:
    """  shifts and rescales the data, but does not change the datashape
    """

    def __init__(self, scale, bias, keys):
        self.bias = bias
        self.scale = scale
        self.keys = keys

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                sequence[key] = sequence[key] * self.scale + self.bias
        return sequence


class Resize:
    """  changes the data dimensions
    """

    def __init__(self, size, keys, mode="nearest"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                sequence[key] = sequence[key].unsqueeze(0)
                sequence[key] = interpolate(
                    sequence[key], size=self.size, mode=self.mode
                )
                sequence[key] = sequence[key].squeeze(0)
        return sequence


class Squeeze:
    """  squeeze a dimension
    """

    def __init__(self, dim=0, keys=None):
        self.dim = dim
        self.keys = keys

    def __call__(self, sequence):
        if self.keys is None:
            keys = sequence.keys()
        else:
            keys = self.keys

        for key in keys:
            if key in sequence.keys():
                sequence[key] = sequence[key].squeeze(self.dim)
        return sequence


class Unsqueeze:
    """  unsqueeze a dimension
    """

    def __init__(self, dim=0, keys=None):
        self.dim = dim
        self.keys = keys

    def __call__(self, sequence):
        if self.keys is None:
            keys = sequence.keys()
        else:
            keys = self.keys

        for key in keys:
            if key in sequence.keys():
                sequence[key] = sequence[key].unsqueeze(self.dim)
        return sequence
