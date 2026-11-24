import torch
from torch import nn
from torch.nn import functional as F

class SpectralConv(nn.Module):

    def __init__(self, width=8, modes=16):
        super().__init__()
        self.modes = modes
        self.positive = nn.Parameter(torch.randn(width, width, modes, modes, dtype=torch.cfloat) / width)
        self.negative = nn.Parameter(torch.randn(width, width, modes, modes, dtype=torch.cfloat) / width)

    def forward(self, x):
        (b, c, h, w) = x.shape
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        out = torch.zeros_like(freq)
        m = min(self.modes, h // 2)
        n = min(self.modes, w // 2 + 1)
        out[:, :, :m, :n] = torch.einsum('bixy,ioxy->boxy', freq[:, :, :m, :n], self.positive[:, :, :m, :n])
        out[:, :, -m:, :n] = torch.einsum('bixy,ioxy->boxy', freq[:, :, -m:, :n], self.negative[:, :, :m, :n])
        return torch.fft.irfft2(out, s=(h, w), norm='ortho')

class FocalPhaseFourierRefiner(nn.Module):

    def __init__(self, width=8, modes=16, blocks=2):
        super().__init__()
        self.lift = nn.Conv2d(6, width, 1)
        self.spectral = nn.ModuleList([SpectralConv(width, modes) for _ in range(blocks)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, current, previous=None):
        with torch.autocast(device_type=current.device.type, enabled=False):
            current = current.float()
            previous = current if previous is None else previous.float()
            (b, h, w) = current.shape
            (y, x) = torch.meshgrid(torch.linspace(-1, 1, h, device=current.device), torch.linspace(-1, 1, w, device=current.device), indexing='ij')
            features = torch.stack([current.cos(), current.sin(), previous.cos(), previous.sin(), x.expand(b, -1, -1), y.expand(b, -1, -1)], 1)
            z = self.lift(features)
            for (spectral, local) in zip(self.spectral, self.local):
                z = z + F.gelu(spectral(z) + local(z))
            delta = torch.pi * torch.tanh(self.head(z)[:, 0])
            return current + delta
