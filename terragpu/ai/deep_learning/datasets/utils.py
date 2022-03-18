import torch
from torch import Tensor
from typing import Any, Dict, List, Sequence


def _dict_list_to_list_dict(sample: Dict[Any, Sequence[Any]]) -> List[Dict[Any, Any]]:
    """Convert a dictionary of lists to a list of dictionaries.
    Args:
        sample: a dictionary of lists
    Returns:
        a list of dictionaries
    .. versionadded:: 0.2
    https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/utils.py
    """
    uncollated: List[Dict[Any, Any]] = [
        {} for _ in range(max(map(len, sample.values())))
    ]
    for key, values in sample.items():
        for i, value in enumerate(values):
            uncollated[i][key] = value
    return uncollated


def unbind_samples(sample: Dict[Any, Sequence[Any]]) -> List[Dict[Any, Any]]:
    """Reverse of :func:`stack_samples`.
    Useful for turning a mini-batch of samples into a list of samples. These individual
    samples can then be plotted using a dataset's ``plot`` method.
    Args:
        sample: a mini-batch of samples
    Returns:
         list of samples
    .. versionadded:: 0.2
    https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/utils.py
    """
    for key, values in sample.items():
        if isinstance(values, Tensor):
            sample[key] = torch.unbind(values)
    return _dict_list_to_list_dict(sample)