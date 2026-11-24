import math
import torch
from torch import nn
from torch.nn import functional as F
from cnn_frontend import PaperCNN
from spectral_phase import FocalPhaseFourierRefiner
from mamba_mixer import SpatialMambaMixer

class CNNFieldCausalTransformer(nn.Module):

    def __init__(self, size=256, patch=8, width=32, heads=4, memory_grid=4, history_frames=20):
        super().__init__()
        if size % patch or width % 4 or width % heads or (history_frames < 1):
            raise ValueError('Invalid dimensions')
        self.size = size
        self.patch = patch
        self.grid = size // patch
        self.width = width
        self.memory_grid = memory_grid
        self.history_frames = history_frames
        if not 1 <= memory_grid <= self.grid:
            raise ValueError('Invalid memory grid')
        self.cnn = PaperCNN()
        self.phase_refiner = FocalPhaseFourierRefiner()
        self.embedding = nn.Linear(2 * patch * patch, width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.spatial = SpatialMambaMixer(width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ff = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, width))
        self.decoder = nn.Linear(width, 2 * patch * patch)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        (y, x) = torch.meshgrid(torch.arange(self.grid), torch.arange(self.grid), indexing='ij')
        freq = torch.exp(torch.arange(width // 4) * (-math.log(10000) / (width // 4)))
        pos = torch.cat([torch.sin(x.flatten()[:, None] * freq), torch.cos(x.flatten()[:, None] * freq), torch.sin(y.flatten()[:, None] * freq), torch.cos(y.flatten()[:, None] * freq)], -1) * 0.1
        self.register_buffer('spatial_position', pos[None])

    def encode_field(self, field, index):
        (b, c, h, w) = field.shape
        p = self.patch
        g = self.grid
        if (c, h, w) != (2, self.size, self.size):
            raise ValueError('Field dimensions mismatch')
        amp = field[:, 0]
        phase = field[:, 1]
        scale = amp.square().mean((-2, -1), keepdim=True).clamp_min(1e-08).sqrt().detach()
        normalized = torch.stack([amp / scale, phase / math.pi], 1)
        tiles = normalized.reshape(b, 2, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(b, g * g, 2 * p * p)
        time = 0.05 * torch.sin(torch.arange(self.width, device=field.device, dtype=field.dtype) * 0.09 + index * 0.2)
        return self.embedding(tiles) + self.spatial_position + time

    def field_to_memory(self, output_field, index):
        tokens = self.encode_field(output_field, index)
        b = output_field.shape[0]
        g = self.grid
        intensity = output_field[:, 0:1].clamp_min(0).square()
        mass = F.avg_pool2d(intensity, self.patch, self.patch)
        mass = mass / mass.mean((-2, -1), keepdim=True).clamp_min(1e-08)
        weight = 0.01 + mass
        values = tokens.transpose(1, 2).reshape(b, self.width, g, g)
        pooled = F.adaptive_avg_pool2d(values * weight, self.memory_grid)
        pooled = pooled / F.adaptive_avg_pool2d(weight, self.memory_grid).clamp_min(1e-08)
        return pooled.flatten(2).transpose(1, 2)

    def step(self, frame, state=None, return_cnn=False):
        if frame.ndim != 4 or tuple(frame.shape[1:]) != (2, self.size, self.size):
            raise ValueError('Expected [B,2,H,W]')
        if state is not None and state['batch_size'] != frame.shape[0]:
            raise ValueError('Reset state when batch changes')
        index = 0 if state is None else state['index']
        (ca, cp) = self.cnn(frame)
        current_field = torch.stack([ca, cp], 1)
        z = self.encode_field(current_field, index)
        if state is not None:
            memory = torch.cat(state['memory_tokens'], 1)
            z = z + self.cross(self.norms[0](z), self.norms[1](memory), memory, need_weights=False)[0]
        n = self.norms[2](z)
        z = z + self.spatial(n, n, z, need_weights=False)[0]
        z = z + self.ff(self.norms[3](z))
        b = frame.shape[0]
        g = self.grid
        p = self.patch
        correction = self.decoder(z).reshape(b, g, g, 2, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 2, self.size, self.size)
        final_field = current_field + correction
        previous = None if state is None else state['output_fields'][-1][:, 1]
        refined_phase = self.phase_refiner(final_field[:, 1], previous)
        final_field = torch.stack([final_field[:, 0].float(), refined_phase], 1)
        old_fields = [] if state is None else state['output_fields']
        old_tokens = [] if state is None else state['memory_tokens']
        new_state = dict(index=index + 1, batch_size=b, output_fields=(old_fields + [final_field])[-self.history_frames:], memory_tokens=(old_tokens + [self.field_to_memory(final_field, index)])[-self.history_frames:])
        outputs = (final_field[:, 0], final_field[:, 1])
        if return_cnn:
            return (outputs, new_state, (ca, cp))
        return (outputs, new_state)

    def forward(self, sequence, return_cnn=False):
        if sequence.ndim != 5 or tuple(sequence.shape[2:]) != (2, self.size, self.size):
            raise ValueError('Expected [B,T,2,H,W]')
        state = None
        amplitude = []
        phase = []
        cnn_fields = []
        for t in range(sequence.shape[1]):
            if return_cnn:
                ((a, p), state, (ca, cp)) = self.step(sequence[:, t], state, True)
                cnn_fields.append(torch.stack([ca, cp], 1))
            else:
                ((a, p), state) = self.step(sequence[:, t], state)
            amplitude.append(a)
            phase.append(p)
        result = (torch.stack(amplitude, 1), torch.stack(phase, 1))
        return (*result, torch.stack(cnn_fields, 1)) if return_cnn else result
