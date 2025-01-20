from dommel_library.train.losses.elbo import NLLLoss, KLLoss
from dommel_library.train.losses.constraint import ConstraintLoss
from dommel_library.train.losses.loss_aggregate import LossAggregate

import torch.nn as nn  # noqa: F401

import logging

logger = logging.getLogger(__name__)


def loss_factory(**losses):
    """
    Factory method for creating a loss aggregate given a list of losses
    :param losses: dict mapping a name to a loss configuration specified by
    a dict with the keys: 'key', 'target', 'type'
    :return: A LossAggregate() object
    """
    for name, loss_dict in losses.items():
        loss_dict["name"] = name
        if loss_dict["type"] == "NLL":
            loss_dict["loss"] = NLLLoss(**loss_dict.get("args", {}))
        elif loss_dict["type"] == "KL":
            loss_dict["loss"] = KLLoss(**loss_dict.get("args", {}))
        elif loss_dict["type"] == "Constraint":
            # constraint on an other loss defined in `for`
            loss_to_constrain_dict = loss_dict.get("for", None)
            if loss_to_constrain_dict is None:
                logger.info(
                    "Specify a loss to constrain in `for` attribute"
                )
                raise ValueError

            loss_to_constrain_dict.update(
                key="prediction",
                target="expectation",
            )
            loss_to_constrain = loss_factory(**{
                name + "_constraint": loss_to_constrain_dict
            })

            constraint_params = loss_dict.get("args", None)
            if constraint_params is None:
                logger.info(
                    "Trying to create a constraint without setting"
                    " the parameters"
                )
                raise ValueError

            loss_dict["loss"] = ConstraintLoss(
                loss_to_constrain,
                name,
                **loss_dict.get("args", {}),
            )

            # by default don't batch average the Constraint,
            # as this should be handled in loss_to_constrain
            if "batch_average" not in loss_dict.keys():
                loss_dict["batch_average"] = False
        else:
            try:
                loss_module = "nn." + loss_dict["type"] + "Loss"
                module = (
                    loss_module
                    + "(**"  # noqa: W503
                    + repr(loss_dict.get("args",
                                         {"reduction": "sum"}))  # noqa: W503
                    + ")"  # noqa: W503
                )
                loss_dict["loss"] = eval(module)
            except Exception:
                logger.info("Loss type not recognized: " + loss_dict["type"])
                raise ValueError

    return LossAggregate(**losses)
