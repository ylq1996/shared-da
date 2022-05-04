import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x without
        output: batch_size x 1 x h x without
    """
    assert v.dim() ==2
    n, c = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * np.log2(c))