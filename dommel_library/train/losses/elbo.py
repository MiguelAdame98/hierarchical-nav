import torch.nn as nn
from torch.distributions.kl import kl_divergence
from dommel_library.distributions import StandardNormal


class NLLLoss(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

    def forward(self, predicted, expected):
        """
        Computes the negative log likelihood
        :param predicted: should be a pytorch distributions object for the
            predicted distribution that implements the log_prob method
        :param expected: ground truth value
        :return: the negative log likelihood
        """
        log_prob = predicted.log_prob(expected)
        return -log_prob.sum()


class KLLoss(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

    def forward(self, predicted, expected=None):
        """
        KL divergence between the predicted distribution and the expected
            distribution
        :param predicted: pytorch distribution object
        :param expected: pytorch distribution object, if none is provided, a
            standard normal distribution is used
        :return: KL divergence
        """
        if expected is None:
            expected = StandardNormal(
                predicted.mean.shape).to(predicted.device)

        kl = kl_divergence(predicted, expected)
        return kl.sum()
