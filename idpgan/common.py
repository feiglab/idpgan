import numpy as np

import torch
import torch.nn as nn


def get_activation(activation, slope=0.2):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "lrelu":
        return nn.LeakyReLU(slope)
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise KeyError(activation)