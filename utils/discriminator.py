import torch
from torch import nn


def get_fc_discriminator(num_classes, ndf=64):

    return nn.Sequential(
        nn.Linear(2, 4),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Linear(4, 8),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Linear(8, 1)


    )

