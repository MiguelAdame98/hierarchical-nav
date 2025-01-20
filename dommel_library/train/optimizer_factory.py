import torch.optim  # noqa: F401


def optimizer_factory(parameters, type, lr, **kwargs):
    """
    Factory method to construct an optimizer
    :param parameters: Trainable parameters
    :param optimizer_type: Name of the optimizer as defined in torch.optim
    :param lr: Learning rate for the chosen optimizer
    :return: an optimizer object
    """
    optimizer = eval("torch.optim." + type + "(parameters, lr=lr, **kwargs)")
    return optimizer
