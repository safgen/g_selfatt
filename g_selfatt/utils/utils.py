import torch
import collections.abc as container_abcs 
from itertools import repeat

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


# to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
# to_3tuple = _ntuple(3)
# to_4tuple = _ntuple(4)
# to_ntuple = _ntuple

def normalize_tensor_one_minone(
    pp: torch.Tensor,
) -> torch.Tensor:
    """
    Receives a tensor and retrieves a normalized version of it between [-1, ..., 1].
    We achieve this by normalizing within [0, 2] and subtracting 1 from the result.
    """
    pp_norm = pp.float()
    pp_max = pp_norm.max() / 2.0
    pp_min = pp_norm.min()

    return (pp_norm / (pp_max - pp_min)) - 1.0


def normalize_tensor(
    pp: torch.Tensor,
) -> torch.Tensor:
    """
    Receives a tensor and retrieves a normalized version of it between [0, ..., 1].
    """
    pp_norm = pp.float()
    pp_max = pp_norm.max()
    pp_min = pp_norm.min()

    return pp_norm / (pp_max - pp_min)


def num_params(
    model: torch.nn.Module,
) -> int:
    """
    Calculates the number of parameters of a torch.nn.Module.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
