import torch
from dommel_library.datastructs import TensorDict
from dommel_library.distributions.multivariate_normal import MultivariateNormal


def multivariate_distribution(mean: torch.Tensor, stdev: torch.Tensor=None) -> MultivariateNormal :
    if stdev is None:
        multivariate = MultivariateNormal(mean)
    else:
        multivariate = MultivariateNormal(mean, stdev)

    return multivariate

def tensor_dict(input: dict) -> TensorDict :
    return TensorDict(input)
