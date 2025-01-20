import math
import importlib
from numbers import Number

from torch import nn
from torch.nn import functional as F

from dommel_library.nn import get_activation, MLP
from dommel_library.nn import ConvFiLM


import matplotlib.pyplot as plt
import torch

class VisualizeLayer(nn.Module):

    def __init__(self, filename):
        nn.Module.__init__(self)
        self.filename = filename
        self.calib = False

    def forward(self, x):
        if not self.calib:
            b, c, h, w = x.shape
            print(x.shape)
            if (c > 7):
                fig, ax = plt.subplots(c//8, 8, figsize=(8 * 2, c//8 * 2))
                [a.axis('off') for a in ax.flatten()]
                if x.type() == 'torch.quantized.QUInt8Tensor':
                    x_p = torch.int_repr(x).numpy()
                    for bi in range(c//8):
                        for ci in range(8):
                            if c//8 == 1:
                                ax[ci].imshow(x_p[0, ci])
                                ax[ci].set_title(f"Ch: {ci}")
                            else:
                                ax[bi, ci].imshow(x_p[0, 8*bi + ci])
                                ax[bi, ci].set_title(f"Ch: {8*bi + ci}")
                    plt.savefig(self.filename, bbox_inches="tight")
                else:
                    for bi in range(c//8):
                        for ci in range(8):
                            if c//8 == 1:
                                ax[ci].imshow(x[0, ci])
                                ax[ci].set_title(f"Ch: {ci}")
                            else:
                                ax[bi, ci].imshow(x[0, 8*bi + ci])
                                ax[bi, ci].set_title(f"Ch: {8*bi + ci}")
                    plt.savefig(self.filename, bbox_inches="tight")
            else:
                fig, ax = plt.subplots(1, 2, figsize=(2 * 2, 1 * 2))
                [a.axis('off') for a in ax.flatten()]
                if x.type() == 'torch.quantized.QUInt8Tensor':
                    x_p = torch.int_repr(x).numpy()
                    for ci in range(2):
                        ax[ci].imshow(x_p[0, ci])
                        ax[ci].set_title(f"Ch: {ci}")
                else:
                    for ci in range(2):
                        ax[ci].imshow(x[0, ci])
                        ax[ci].set_title(f"Ch: {ci}")
                plt.savefig(self.filename, bbox_inches="tight")
            plt.show()
        return x

class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation="Activation",
        batch_norm=False,
        condition_size=0,
        **kwargs
    ):
        """
        A basic convolutional building block consisting of a convolution with
        an activation, and optionally batch normalization and FiLM conditioning
        :param in_channels:  Length of the input channel.
        :param out_channels: Length of the output channel.
        :param activation: String indicating the activation function.
        :param batch_norm: Boolean to indicate whether to use batch norm.
        :param condition_size: Optional length for the FiLM conditioning.
        :param kwargs: Parameters to pass to conv2d from torch.
        """
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = kwargs.get("kernel_size", 3)
        stride = kwargs.get("stride", 1)
        padding = kwargs.get("padding", (kernel_size - 1) // 2)
        bias = kwargs.get("bias", True)

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias)
        self.act = get_activation(activation, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        if condition_size > 0:
            self.film = ConvFiLM(condition_size, out_channels)
        else:
            self.film = None

    def forward(self, x, condition=None):
        y = self.conv(x)
        # a = VisualizeLayer('/home/idlab332/workspace/active-inference/filename_'+ str(self.in_channels)+'_'+ str(self.out_channels) )
        # a.forward(y)
        if self.batch_norm:
            y = self.batch_norm(y)
        if self.film:
            if condition is None:
                raise Exception("Condition input expected!")
            y = self.film(y, condition)
        y = self.act(y)
        return y


class ConvPipeline(nn.Module):
    """Builds a conv layer sequence that downscales the input tensor
    according to given channels, it will reshape the output to a vector
    """

    def __init__(
        self,
        input_shape,
        channels,
        block="Conv",
        kernel_size=None,
        stride=None,
        activation="Activation",
        batch_norm=False,
        condition_size=0,
        flatten=True,
        **kwargs
    ):
        """Creates a convolutional pipeline of conv blocks.
        The optional arguments for stride and kernel_size can be provided
        as either a single element -- meaning that all layers should share
        the same parameter -- or as a list, indicating that these values
        should be used. In that case a value must be provided for all layers.
        If no value is provided, a default value will be taken.
        The default kernel_size is 3.
        The default stride is 1 if subsequent channels are the same, or 2 if
        the number of channels changes.
        :param input_shape: The shape of the input. Must be CxHxW.
        :param channels: List of channel lengths the pipeline must go through.
        :param block: The type of block used, can also be a list.
        :param kernel_size: Optional list of kernel sizes.
        :param stride: Optional list of strides.
        :param activation: The activation function to be used.
        :param batch_norm: Flag to indicate the usage of batch norm.
        :param condition_size: Whether to use a film conditioning.
        :param flatten: Flatten the output to a single dimension vector.
        :param kwargs: Contains optional Conv block parameters.
        """
        nn.Module.__init__(self)
        self.input_shape = input_shape
        self.flatten = flatten

        self.channels = _parse_input_list(channels)
        self.blocks = _parse_input_list(block, None, len(channels))
        self.kernel_sizes = _parse_input_list(kernel_size, 3, len(channels))
        for k in self.kernel_sizes:
            if k % 2 == 0:
                raise ValueError(
                    "Even kernel sizes are not supported in ConvPipeline")

        self.strides = _parse_input_list(stride, None, len(channels))
        self.activations = _parse_input_list(activation, "Activation",
                                             len(channels))
        if "padding" in kwargs.keys():
            raise ValueError(
                "Custom padding is not supported in ConvPipeline")

        self.layers = nn.ModuleList()

        self.output_shape = input_shape[:]
        in_channels = input_shape[0]
        for i in range(len(self.channels)):
            block = self.blocks[i]
            b = block.split(".")
            if len(b) > 1:
                package = '.'.join(b[:-1])
                block = b[-1]
                block_module = importlib.import_module(package)  # noqa: F
                block = '.'.join(["block_module", block])

            out_channels = self.channels[i]
            kernel_size = self.kernel_sizes[i]
            stride = self.strides[i]
            if not stride:
                stride = 1 if in_channels == out_channels else 2
            activation = self.activations[i]
            module = (
                f"{block}("
                f"in_channels={in_channels},"
                f"out_channels={out_channels},"
                f"kernel_size={kernel_size},"
                f"stride={stride},"
                f"activation={repr(activation)},"
                f"batch_norm={batch_norm},"
                f"condition_size={condition_size},"
                f"**{kwargs})"
            )
            self.layers.append(eval(module))
            

            # adjust output shape
            self.output_shape[0] = out_channels
            self.output_shape[1] = math.ceil(self.output_shape[1] / stride)
            self.output_shape[2] = math.ceil(self.output_shape[2] / stride)

            # update in_channels for next
            in_channels = out_channels

        self.output_length = int(
            self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        )

    def forward(self, x, condition=None):
        y = x
        for layer in self.layers:
            y = layer(y, condition=condition)
        if self.flatten:
            y = y.reshape(-1, self.output_length)
        return y


class Interpolate(nn.Module):
    """Module for upsampeling a tensor in it's spacial dimensions.
    Provides a wrapper around torch.nn.F.interpolate .
    """

    def __init__(self, scale_factor=2, mode="nearest", **kwargs):
        """ Initializes the Interpolation step
        :param scale_factor: How much to upscale.
        :param mode: Interpolation method.
        """
        nn.Module.__init__(self)
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x, *other):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class UpConvPipeline(nn.Module):
    def __init__(
        self,
        output_shape,
        channels,
        block="Conv",
        kernel_size=None,
        interpolate=None,
        activation="Activation",
        batch_norm=False,
        condition_size=0,
        **kwargs
    ):
        """Creates an upsampeling conv pipeline, provides the opposite
        operation of a ConvPipeLine, if instantiated with the same parameters
        :param output_shape: The shape of the desired output. Must be CxHxW.
        :param channels: List of channel lengths the pipeline must go through.
        :param block: The type of block used, can also be a list.
        :param kernel_size: Optional list of kernel sizes.
        :param activation: The activation function to be used.
        :param interpolate: Optional list of interpolate factors.
        :param batch_norm: Flag to indicate the usage of batch norm.
        :param condition_size: Whether to use a film conditioning.
        :param kwargs: Contains optional Conv block parameters.
        """
        nn.Module.__init__(self)
        self._output_shape = output_shape

        self.channels = _parse_input_list(channels)
        self.blocks = _parse_input_list(block, None, len(channels))
        self.kernel_sizes = _parse_input_list(kernel_size, 3, len(channels))
        for k in self.kernel_sizes:
            if k % 2 == 0:
                raise ValueError(
                    "Even kernel sizes are not supported in UpConvPipeline")

        self.interpolate = _parse_input_list(interpolate, None, len(channels))
        self.activations = _parse_input_list(activation, "Activation",
                                             len(channels))
        if "padding" in kwargs.keys():
            raise ValueError(
                "Custom padding is not supported in UpConvPipeline")

        # calculate expected input shape
        in_channels = self.channels[0]
        height = output_shape[1]
        width = output_shape[2]
        #print('output height and width', height, width)
        channels.append(output_shape[0])

        self.layers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            block = self.blocks[i]
            b = block.split(".")
            if len(b) > 1:
                package = '.'.join(b[:-1])
                block = b[-1]
                block_module = importlib.import_module(package)  # noqa: F
                block = '.'.join(["block_module", block])

            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]

            # check if we need to interpolate
            if not self.interpolate[i]:
                if in_channels != out_channels:
                    self.interpolate[i] = 2
                else:
                    self.interpolate[i] = 1
            if self.interpolate[i] > 1:
                mode = kwargs.get('interpolate_mode', 'nearest')
                self.layers.append(Interpolate(self.interpolate[i], mode))

            height = math.ceil(height / self.interpolate[i])
            width = math.ceil(width / self.interpolate[i])
            #print('updated height and width', in_channels, out_channels, height, width)

            kernel_size = self.kernel_sizes[i]
            activation = self.activations[i]
            module = (
                f"{block}("
                f"in_channels={in_channels},"
                f"out_channels={out_channels},"
                f"kernel_size={kernel_size},"
                f"stride=1,"
                f"activation={repr(activation)},"
                f"batch_norm={batch_norm},"
                f"condition_size={condition_size},"
                f"**{kwargs})"
            )
            self.layers.append(eval(module))

            # update in_channels for next
            in_channels = out_channels

        self.reshape_shape = [channels[0], height, width]
        #print('FINAL input height and width', height, width)

    def forward(self, z, condition=None):
        z = z.reshape(-1, *self.reshape_shape)
        for layer in self.layers:
            z = layer(z, condition)
        if z.shape[-1] != self._output_shape[-1]:
            z = z.narrow(-1, 0, self._output_shape[-1])
        if z.shape[-2] != self._output_shape[-2]:
            z = z.narrow(-2, 0, self._output_shape[-2])
        return z


class CNN(nn.Module):
    """Convenience wrapper around ConvPipeLine and MLP"""

    def __init__(
        self,
        input_shape,
        output_length,
        channels,
        hidden=None,
        activation="Activation",
        **kwargs
    ):
        """ Builds the CNN
         :param input_shape: The input shape in CxHxW format.
         :param output_length: The desired output vector length.
         :param channels: List of channel lengths the pipeline must go through.
         :param hidden: Optional list of hidden neuron lengths for the MLP.
         :param activation: The activation function for the pipeline
         :param kwargs: Optional parameters for the Conv2d and activation.
         """
        nn.Module.__init__(self)
        self._act = get_activation(activation, **kwargs)
        self._convs = ConvPipeline(
            input_shape, channels, activation=activation, **kwargs
        )
        self._fc = MLP(
            self._convs.output_length, output_length, hidden, activation
        )

    def forward(self, x):
        h = self._act(self._convs(x))
        h.reshape(-1, self._convs.output_length)
        y = self._fc(h)
        return y


class UpCNN(nn.Module):
    """Convenience wrapper around UpConvPipeline and MLP"""

    def __init__(
        self,
        input_length,
        output_shape,
        channels,
        hidden=None,
        activation="Activation",
        **kwargs
    ):
        """Initializes the InvConv
        :param input_length: The length of the input vector.
        :param output_shape: The desired output shape in CxHxW format.
        :param channels: The channels the upsampling must go through.
        :param hidden: optional list of hidden neurons of the MLP.
        :param activation: The activation function to use. Defaults to
        leaky_relu.
        :param kwargs: Optional parameters for the activation and conv2d
        """
        nn.Module.__init__(self)
        self._act = get_activation(activation, **kwargs)

        if hidden is not None:
            size = hidden.pop()
            self._fc = MLP(input_length, size, hidden, activation, **kwargs)
        else:
            # no linear layer size is provided
            # So just reshape when assuming doubling every layer
            c = channels[0]
            h = output_shape[1] // 2 ** len(channels)
            w = output_shape[2] // 2 ** len(channels)
            self._fc = nn.Linear(input_length, h * w * c)

        self._iconvs = UpConvPipeline(
            output_shape, channels, activation=activation, **kwargs
        )

    def forward(self, x):
        h = self._act(self._fc(x))
        y = self._iconvs(h)
        return y


def _parse_input_list(li, default=None, target_length=None):
    if not li and not isinstance(li, Number):
        # use default when li is None or empty string
        # but not when 0
        li = [default] * target_length
    elif isinstance(li, str):
        if "*" in li or "+" in li:
            # string contains operators, eval
            li = eval(li)
        else:
            # we want a list of strings, expand
            li = [li] * target_length
    elif not hasattr(li, "__iter__"):
        # single value, expand
        li = [li] * target_length
    elif target_length:
        # check for ellipsis
        for i in range(len(li)):
            if li[i] is Ellipsis or li[i] == "...":
                # replace and repeat item before or after
                if i > 0:
                    li = li[0:i] + [li[i - 1]] * \
                        (target_length + 1 - len(li)) + li[i + 1:]
                else:
                    li = [li[1]] * (target_length + 1 - len(li)) + li[1:]
                    break

        if len(li) != target_length:
            raise AssertionError(
                "When providing a list, it should"
                "contain as much entries as layers"
            )
    return li
