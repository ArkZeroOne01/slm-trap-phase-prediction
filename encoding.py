import torch

def ifftc(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

def encode(q, phase, size):
    (b, n, _) = q.shape
    xy = q + size // 2
    ij = xy.floor().long()
    frac = xy - ij
    if ((ij < 0) | (ij >= size - 1)).any():
        raise ValueError('Trap outside input grid')
    amp = torch.zeros(b, size * size, device=q.device)
    ph = torch.zeros_like(amp)
    occupancy = torch.zeros_like(amp)
    for (dx, dy) in ((0, 0), (1, 0), (0, 1), (1, 1)):
        index = (ij[..., 1] + dy) * size + ij[..., 0] + dx
        weight = (frac[..., 0] if dx else 1 - frac[..., 0]) * (frac[..., 1] if dy else 1 - frac[..., 1])
        amp.scatter_add_(1, index, weight)
        ph.scatter_(1, index, phase)
        occupancy.scatter_add_(1, index, torch.ones_like(phase))
    if occupancy.max() > 1:
        raise ValueError('Overlapping input phase pixels: ambiguous SM rasterization')
    return torch.stack((amp, ph), 1).reshape(b, 2, size, size)

def to_mask(amp, phase):
    return ifftc(torch.polar(amp.float().clamp_min(0), phase.float())).angle()
