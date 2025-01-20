import torch

from dommel_library.datastructs.tensor_dict import TensorDict


class ComposableModule(torch.nn.Module):
    def __init__(self, modules):
        """
        :param modules: a list of modules that are used in the composable
            model. Each list element is a dictionary that describes the module
            using the following keys ["input", "output", "module"]
        """
        torch.nn.Module.__init__(self)

        self._module_list = modules

        # register modules for parameters(), to() and autograd
        for i, module in enumerate(self._module_list):
            self.add_module(f"module-{i}", module["module"])

    def forward(self, input_dict):
        """
        :param input_dict:
        :return: output_dict
        """
        output_dict = {}

        x = None
        for module in self._module_list:
            # Tries to get the value from the input dict, if it's not there,
            # it searches for the value in the output dict (from a previously
            # computed module), If that's not there, it takes the previously
            # computed value, as defined in last layer.

            module_args = []

            # Get input values
            input_keys = module.get("input", None)
            if input_keys is not None:
                # make it a list: unpacking can be done when calling the module
                if not isinstance(input_keys, list):
                    input_keys = [input_keys]
                for key in input_keys:
                    # add previous outputs using the "..." key
                    if key == "...":
                        if isinstance(x, tuple):
                            module_args += list(x)
                        else:
                            module_args.append(x)
                    else:
                        val = input_dict.get(key, output_dict.get(key, None))
                        module_args.append(val)
            else:
                if x is None:
                    module_type = module["type"]
                    message = (f"No valid input provided for {module_type}. "
                               f"Check your configuration for input keys.")
                    raise Exception(message)
                elif type(x).__name__ == "QuantTensor":
                    module_args.append(x)
                elif isinstance(x, tuple):
                    module_args += list(x)
                else:
                    module_args.append(x)

            # compute forward pass
            x = module["module"](*module_args)

            # store output of module in the dictionary of outputs
            output_keys = module.get("output", None)
            if output_keys is not None:
                if isinstance(output_keys, list):
                    for i, item in enumerate(x):
                        output_dict[output_keys[i]] = item
                else:
                    output_dict[output_keys] = x

        return TensorDict(output_dict)
