from dommel_library.train.losses.loss import Loss
from dommel_library.train.losses.log import Log


class LossAggregate(Loss):
    def __init__(self, **losses):
        """
        :param losses: dict of losses dictionaries.
            required keys: ['key', 'target', 'type', 'name']
            optional keys: ['weight']
            additional keys specific to the loss function can be provided.
        """
        self._losses = losses
        self._total_loss = 0
        self._log_values = {}

    def __call__(self, predictions, expectations):
        """
        Implements the loss function
        :param predicted: Dictionary of predicted tensors
        :param expected: Dictionary of ground_truth tensors
        :return: loss value on which .backward() can be called
        """
        self._total_loss = 0
        for _, loss_dict in self._losses.items():
            p_key = loss_dict["key"]
            t_key = loss_dict["target"]
            loss_name = loss_dict["name"]
            # print('loss aggegate p and t keys', p_key, t_key)
            # print(predictions.keys(), expectations.keys())
            prediction = predictions.get(p_key, None)
            target = expectations.get(t_key, predictions.get(t_key, None))

            # TODO: pass **kwargs, does not work for MSE
            loss_value = loss_dict["loss"](prediction, target)
            batch_average = loss_dict.get("batch_average", True)
            if batch_average:
                if len(predictions.shape) > 1:
                    loss_value = loss_value / predictions.shape[0]

            self._log_values[f"Loss/{loss_name}"] = loss_value

            weight = loss_dict.get("weight", 1)
            self._total_loss += weight * loss_value

        return self._total_loss

    def post_backprop(self):
        for _, loss_dict in self._losses.items():
            # will fail if the function is not a dommel Loss
            try:
                loss_dict["loss"].post_backprop()
            except AttributeError:
                continue

    def parameters(self):
        trainable_parameters = []
        for _, loss_dict in self._losses.items():
            # will fail if the function is not a dommel Loss
            try:
                trainable_parameters += loss_dict["loss"].parameters()
            except AttributeError:
                continue
        return trainable_parameters

    @property
    def logs(self):
        """
        :return: the logs for all components in this loss function
        """
        log = Log()
        # log all separate elements of loss
        for k, v in self._log_values.items():
            log.add(k, v.detach().cpu().numpy(), "scalar")
        log.add("Loss/loss", self._total_loss.detach().cpu().numpy(), "scalar")
        # If the implemented losses have additional logs
        for _, loss_dict in self._losses.items():
            # will fail if the function is not a dommel Loss
            try:
                log += loss_dict["loss"].logs
            except AttributeError:
                continue
        return log

    def to(self, *args, **kwargs):
        """
        Move all sub models to the desired device
        """
        for k, loss_dict in self._losses.items():
            try:
                # will fail if the loss does not have a .to() method
                loss_dict["loss"] = loss_dict["loss"].to(*args, **kwargs)
            except AttributeError:
                continue
        return self
