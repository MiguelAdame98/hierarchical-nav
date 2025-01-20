# Losses 

The `dommel.losses.loss_factory.loss_factory(loss_dicts)` method can be used to create an aggregate loss function from a list of dictionaries describing the loss. 
Each loss dictionary should have the following keys: 
required:
- `type`: The type of loss, can be `MSE`, `KL`, `NLL` or `Constraint`
- `key`: contains the name of the predicted tensor
- `target`: contains the name of the target ground truth tensor
optional: 
- `weight`: weight of a loss term in the global loss
Only if `type` == `Constraint`:
- `constraint_parameters`: A dictionary containing the keyword arguments of the `Constraint` class, in order to create this directly from the config file. 
- `reconstruction`: The loss that should be used for reconstruction in the constraint. Can be `MSE` or `NLL`.

An example for this use is shown in `examples/mnist_vae.py` and `examples/mnist_ae.py`.

The loss for GECO can be defined as: 
```yaml
loss:
    - type: KL
      key: posterior 
      value: std_normal 
    - type: Constraint
      key: image
      value: image 
      reconstruction: NLL
      constraint_parameters: 
          tolerance: 180
```

___
A custom method for defining loss functions that can work together with the `Trainer` class. 
The loss function is able to compute a loss value and provide logs in the desired format to the trainer. 

The `CustomLoss` should therefore:
 - inherit from `Loss`
 - implement the `__call__` method which computes the loss value. 
 - By default the `logs` property will return an empty log object. A log object is used to log to tensorboard. A scalar should have the type `"scalar"` and an image the type `"image"`. 
 - The `trainable_parameters` property will return an empty list by default, if the loss functions should use the gradient, it should override this propety, as well as the `post_backprop()` method.

