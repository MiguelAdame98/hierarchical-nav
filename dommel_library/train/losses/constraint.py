import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from dommel_library.train.losses import Loss, Log

from dommel_library.datastructs.tensor_dict import TensorDict


class ConstraintLoss(Loss):
    def __init__(self, internal_loss, name, **constraint_dict):
        # name for logging
        self._name = name

        self._internal_loss = internal_loss
        self._constraint = Constraint(**constraint_dict)

        self._error = None
        self._last_value = None

    def __call__(self, prediction, expectation):
        self._error = self._internal_loss(
            TensorDict({"prediction": prediction}),
            TensorDict({"expectation": expectation}),
        )
        # [0] because from tensor of dim 1 to dim 0
        self._last_value = self._constraint(self._error)[0]
        return self._last_value

    @property
    def logs(self):
        log = Log()
        log.add(
            f"{self._name}/unconstrained",
            self._error.cpu().detach().numpy(),
            "scalar",
        )
        log.add(
            f"{self._name}/constrained",
            self._last_value.cpu().detach().numpy(),
            "scalar",
        )
        log.add(
            f"{self._name}/lambda",
            self._constraint.multiplier.cpu().detach().numpy(),
            "scalar",
        )
        log.add(
            f"{self._name}/tolerance",
            self._constraint.tolerance,
            "scalar",
        )
        return log

    def parameters(self):
        return list(self._constraint.parameters())

    def post_backprop(self):
        self._constraint.adjust_tolerance()
        try:
            self._internal_loss.post_backprop()
            # will fail if not a dommel loss
        except AttributeError:
            pass

    def to(self, *args, **kwargs):
        self._constraint.device = args[0]
        self._constraint.lambd_min = self._constraint.lambd_min.to(
            *args, **kwargs
        )
        self._constraint.lambd_max = self._constraint.lambd_max.to(
            *args, **kwargs
        )
        self._constraint.multiplier = self._constraint.multiplier.to(
            *args, **kwargs
        )
        self._constraint = self._constraint.to(*args, **kwargs)
        return self


class Constraint(nn.Module):
    def __init__(
        self,
        tolerance,
        lambda_min=0.0,
        lambda_max=20.0,
        lambda_init=1.0,
        alpha=0.99,
        adjust_tolerance=False,
        adjust_frequency=1000,
        adjust_tangent_threshold=0.2,
        adjust_tolerance_factor=1.01,
        device="cpu",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.moving_average = None
        self.tolerance = tolerance
        self.device = device
        self.lambd_min = torch.tensor([lambda_min], dtype=torch.float).to(
            self.device
        )
        self.lambd_max = torch.tensor([lambda_max], dtype=torch.float).to(
            self.device
        )
        self.multiplier = torch.tensor([lambda_init], dtype=torch.float).to(
            self.device
        )
        self.lambd = nn.Parameter(self._inv_squared_softplus(self.multiplier))
        self.alpha = alpha
        self.clamp = ClampFunction()
        self.prev_grad = None
        self.last_grad = None
        self.delta = adjust_frequency
        self.tangent = 0.0
        self.tangent_threshold = adjust_tangent_threshold
        self.tolerance_factor = adjust_tolerance_factor
        self.tolerance_fixed = not adjust_tolerance
        self.i = 0

        # variables required for online sleep
        self.constraint_is_hit = False
        self.constraint_has_been_hit = False

    def forward(self, value):
        constraint = value - self.tolerance
        self.constraint_is_hit = constraint < 0
        self.constraint_has_been_hit = self.constraint_has_been_hit or (
            constraint < 0)

        with torch.no_grad():
            if self.moving_average is None:
                self.moving_average = constraint
            else:
                self.moving_average = (
                    self.alpha * self.moving_average
                    + (1 - self.alpha) * constraint  # noqa: W503, W504
                )

        cost = constraint + (self.moving_average - constraint).detach()

        # we use squared softplus as in
        # https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules
        # /optimization_constraints.py
        # we also clamp the resulting values
        self.multiplier = self.clamp.apply(
            F.softplus(self.lambd) ** 2, self.lambd_min, self.lambd_max
        )

        return self.multiplier * cost

    def get_multiplier(self):
        return self.multiplier.item()

    def _inv_squared_softplus(self, x):
        sqrt = torch.sqrt(x)
        return torch.log(torch.exp(sqrt) - 1.0)

    def adjust_tolerance(self):
        # adjust tolerance
        if self.tolerance_fixed:
            return

        if self.i % self.delta == 0:
            if self.prev_grad is None:
                self.prev_grad = self.lambd.grad.item()
            elif self.last_grad is None:
                self.last_grad = self.lambd.grad.item()
            else:
                self.tangent = (self.last_grad - self.prev_grad) / self.delta
                self.prev_grad = self.last_grad
                self.last_grad = self.lambd.grad.item()
                # TODO which threshold and which tolerance adjustment?
                if self.last_grad > 0:
                    # no longer adjust tolerance once the loss reaches the
                    # current tolerance threshold
                    self.tolerance_fixed = True
                elif self.tangent < self.tangent_threshold:
                    self.tolerance *= self.tolerance_factor

        self.i += 1


class ClampFunction(Function):
    """
    Clamp a value between min and max.
    When the gradients push the value further away from the [min,max] range,
    set to zero
    When the gradients push the value back in the [min,max] range,
    let them flow through
    """

    @staticmethod
    def forward(ctx, lambd, min_value, max_value):
        ctx.save_for_backward(lambd, min_value, max_value)
        if lambd < min_value:
            return min_value
        elif lambd > max_value:
            return max_value
        else:
            return lambd

    @staticmethod
    def backward(ctx, lambd_grad):
        lambd, min_value, max_value = ctx.saved_tensors

        if lambd < min_value and lambd_grad < 0.0:
            grad = torch.tensor([0.0], device=lambd_grad.device)
        elif lambd > max_value and lambd_grad > 0.0:
            grad = torch.tensor([0.0], device=lambd_grad.device)
        else:
            grad = -lambd_grad
        return grad, None, None
