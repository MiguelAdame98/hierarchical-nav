import importlib

import torch.nn

# Has to import under a different name to avoid dependency issues with backends
from dommel_library import nn as dnn
from dommel_library.nn.composable_module import ComposableModule

import logging
import inspect

logger = logging.getLogger(__name__)

try:
    import brevitas.nn as qnn
    backends = [dnn, torch.nn, qnn]
except Exception:
    backends = [dnn, torch.nn]


def register_backend(backend):
    """
    Register a backend against the global backends in which is checked for
    modules
    :param backend: the new backend to add
    """
    global backends

    # import backend
    backend_module = importlib.import_module(backend)  # noqa: F
    backends.insert(0, backend_module)


def get_module_from_dict(module_dict):
    module_type = module_dict["type"]
    backend = module_dict.get("backend", None)

    b = module_type.split(".")
    if len(b) > 1:
        backend = '.'.join(b[:-1])
        module_type = b[-1]

    if backend is not None:
        backend_module = importlib.import_module(backend)  # noqa: F
        backend = "backend_module"
    else:
        for i, backend in enumerate(backends):
            if module_type in dir(backend):
                backend = f"backends[{i}]"
                break
        else:
            raise ImportError(
                f"{module_type} not found in "
                f"{[b.__name__ for b in backends]}"
            )
    sig = inspect.signature(eval(f"{backend}.{module_type}")).parameters.keys()
    if "kwargs" in sig:
        module = (
            f"{backend}.{module_type}(**{repr(module_dict.get('args', {}))})"
        )
    else:
        # If module does not take kwargs, parse additional arguments to only
        # pass matching
        module = f"{backend}.{module_type}("
        for k in sig:
            val = module_dict.get("args", {}).get(k, None)
            if val is not None:
                module += f"{k}={repr(val)},"
        module += ")"
    return eval(module)


def module_factory(modules, **kwargs):
    """
    :param modules: a list of dictionaries describing the different modules
        the dictionary should contain the keys ["input", "output", "module"]
        and the "module" should be a dictionary with the keys:
        ["type", "args", "backend"]
    :return: A ComposableModule() from the configuration list
    """
    for module_dict in modules:
        module_dict["module"] = get_module_from_dict(module_dict)
    return ComposableModule(modules)
