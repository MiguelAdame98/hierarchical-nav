from dommel_library.train.losses.log import Log


class Loss:
    def __call__(self, predicted, expected):
        """
        Implements the loss function
        :param predicted: Dictionary of predicted tensors
        :param expected: Dictionary of ground_truth tensors
        :return: loss value on which .backward() can be called
        """
        raise NotImplementedError

    @property
    def logs(self):
        """
        :return: A Log object containing information of the training step
        """
        return Log()

    def parameters(self):
        """
        :return: A list of trainable parameters
        """
        return []

    def post_backprop(self):
        """
        If things need to be done after the backwards pass to the loss function
        e.g. updating tolerances in GECO loss functions...
        :return: nothing
        """
        pass
