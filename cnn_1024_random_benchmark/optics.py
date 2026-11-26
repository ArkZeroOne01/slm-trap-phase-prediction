import torch

def to_mask(amplitude,phase):
    field=torch.polar(amplitude.float().clamp_min(0),phase.float())
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field,dim=(-2,-1)),norm='ortho'),dim=(-2,-1)).angle()
