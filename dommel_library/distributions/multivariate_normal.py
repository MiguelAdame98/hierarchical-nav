import math
import numpy

import torch
from torch.distributions.kl import register_kl

"""
  We use custom classes for distributions that are also torch.Tensor instances
  This is a bit hacky, but allows to handle them nicely in TensorDict
"""


def MultivariateNormal(*t):
    if len(t) == 1:
        t = t[0]
    elif len(t) == 2:
        t = torch.cat(t, -1)

    # this is already a MultivariateNormal distribution, return
    if isinstance(t, InnerMultivariateNormal):
        return t

    # call type only once to make sure .__class__ is equals for objects
    if not hasattr(MultivariateNormal, "cls"):
        cls = type(
            "MultivariateNormal", (InnerMultivariateNormal, t.__class__), {}
        )
        setattr(MultivariateNormal, "cls", cls)

    t.__class__ = MultivariateNormal.cls
    return t


def StandardNormal(*dims):
    mu = torch.zeros(*dims, dtype=torch.float32)
    sigma = torch.ones(*dims, dtype=torch.float32)
    t = torch.cat((mu, sigma), -1)
    return MultivariateNormal(t)


class InnerMultivariateNormal:
    @property
    def variance(self):
        _, sigma = self._mu_sigma()
        return sigma ** 2

    @variance.setter
    def variance(self, value):
        self.stdev = torch.sqrt(value)

    @property
    def covariance_matrix(self):
        return self.variance.unsqueeze(1) * torch.eye(
            self.variance.size(-1)
        ).to(self.variance.device)

    @property
    def mean(self):
        mu, _ = self._mu_sigma()
        return mu

    @mean.setter
    def mean(self, value):
        split = self.shape[-1] // 2
        self[..., :split] = value

    @property
    def stdev(self):
        _, sigma = self._mu_sigma()
        return sigma

    @stdev.setter
    def stdev(self, value):
        split = self.shape[-1] // 2
        self[..., split:] = value

    def sample(self, no_samples=1):
        mu, sigma = self._mu_sigma()
        shape = mu.shape
        if no_samples > 1:
            shape = [no_samples] + list(shape)
        eps = torch.randn(shape).to(mu.device)
        return mu + eps * sigma

    def __mul__(self, other):
        mu1, sigma1 = self._mu_sigma()
        mu2, sigma2 = other._mu_sigma()
        mu = (mu1 * sigma2 ** 2) + (mu2 * sigma1 ** 2)
        mu /= sigma1 ** 2 + sigma2 ** 2
        sigma = sigma1 ** 2 * sigma2 ** 2
        sigma /= sigma1 ** 2 + sigma2 ** 2
        sigma = torch.sqrt(sigma)
        mul = torch.cat((mu, sigma), -1)
        return MultivariateNormal(mul)

    def log_prob(self, value):
        mu, sigma = self._mu_sigma()
        var = sigma ** 2 + 1e-12
        k = int(mu.size()[1])
        if len(sigma.size()) > 2 or len(value.size()) > 2:
            # our input has more than one dimension, vectorize it
            k = numpy.prod(mu.size()[1:])
            var = sigma.reshape(-1, k) ** 2
            mu = mu.reshape(-1, k)
            value = value.reshape(-1, k)

        # use the multivariate gaussian definition
        # we can simplify the determinant to a cumprod due to the fact that
        # sigma is diag(variance**(1/2))
        # we can simplify the (value - mu)^T * sigma^-1 * (value - mu)
        # to (value - mu) ** 2 * sigma (all elementswise) and summing over
        # them note that for sigma = 1 we get a loss similar mse
        # (up to a scalar k log(2*pi))
        term = k * numpy.log(2 * math.pi)
        term = torch.tensor(term).to(mu.device)
        log_det = torch.sum(var.log(), dim=1)
        nll = 0.5 * (
            log_det
            + term
            + torch.sum((value - mu).pow(2) / (var), dim=1)
        )
        return -nll

    # this is untested code!
    def entropy(self):
        _, sigma = self._mu_sigma()
        H = (
            0.5
            + 0.5 * math.log(2 * math.pi)
            + (torch.prod(sigma, dim=1) + 1e-12).log()
        )
        return H

    def _mu_sigma(self):
        split = self.shape[-1] // 2
        mu = self[..., 0:split]
        mu.__class__ = torch.Tensor
        sigma = self[..., split:]
        sigma.__class__ = torch.Tensor
        return mu, sigma

    def to(self, device):
        moved = self.__class__.__bases__[1].to(self, device)
        if isinstance(moved, InnerMultivariateNormal):
            return moved
        return MultivariateNormal(moved)

    def __getitem__(self, index):
        item = self.__class__.__bases__[1].__getitem__(self, index)
        return MultivariateNormal(item)


@register_kl(InnerMultivariateNormal, InnerMultivariateNormal)
def kl_imn_imn(p, q):
    sigma_ratio_squared = (p.stdev / (q.stdev + 1e-12)).pow(2)
    kl = 0.5 * (
        ((p.mean - q.mean) / (q.stdev + 1e-12)).pow(2)
        + sigma_ratio_squared
        - sigma_ratio_squared.log()
        - 1
    )
    return torch.sum(kl, dim=-1)
