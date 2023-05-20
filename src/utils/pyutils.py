import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random state
    """
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
