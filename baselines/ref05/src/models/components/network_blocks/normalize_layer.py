import torch.nn as nn
import torch.nn.functional as F


class NormalizeLayer(nn.Module):
    def __init__(self, dim=-1, p=2):


        super(NormalizeLayer, self).__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)
