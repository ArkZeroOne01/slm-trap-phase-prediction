import torch
from torch import nn

class SpatialMambaMixer(nn.Module):

    def __init__(self, width):
        super().__init__()
        from mamba_ssm import Mamba
        self.ssm = Mamba(d_model=width, d_state=16, d_conv=4, expand=1)
        self.local = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.project = nn.Linear(width, width)

    def forward(self, query, key=None, value=None, need_weights=False, **kwargs):
        return (self.project(self.ssm(query) + self.local(query)), None)
