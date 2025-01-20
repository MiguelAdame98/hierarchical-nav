import torch.nn as nn


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'

    The input data (x) is scaled and a bias is added on a feature level,
    which are computed by a given condition (c).
    """

    def __init__(self, in_shape, out_shape):
        """
        :param in_shape: Shape of the condition c
        :param out_shape: Shape of the transforming data x
        """
        nn.Module.__init__(self)
        self.gamma = nn.Linear(in_shape, out_shape)
        self.beta = nn.Linear(in_shape, out_shape)

    def forward(self, x, c):
        """
        :param x: Input data to transform
        :param c: Condition that defines the transformation
        :return: Transformed data
        """
        v = self.gamma(c) * x.view(x.shape[0], -1) + self.beta(c)
        return v.view(x.shape)


class ConvFiLM(nn.Module):
    """
    In a convolutional layer, the FiLM transform is the same
    over all spatial locations but differs per feature.

    A Linear transformation (scaling and added bias) for each feature channel
    of the output of a convolution x is computed based on a condition value c.
    """

    def __init__(self, condition_shape, n_features):
        """
        :param condition_shape: shape of the condition parameter
        :param n_features: number of feature maps of the convolution output
            that will be transformed.
        """
        nn.Module.__init__(self)
        self.gamma = nn.Linear(condition_shape, n_features)
        self.beta = nn.Linear(condition_shape, n_features)

    def forward(self, x, c):
        """
        :param x: Data to transform
        :param c: Condition on which the transformation is based
        :return: transformed data
        """
        gamma = (
            self.gamma(c)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, x.shape[-2], x.shape[-1])
        )
        beta = (
            self.beta(c)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, x.shape[-2], x.shape[-1])
        )
        return gamma * x + beta
