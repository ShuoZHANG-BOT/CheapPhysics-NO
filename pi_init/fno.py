"""Fourier neural operators used in all experiments.

Adapted from PDEBench (https://github.com/pdebench/PDEBench), which in turn
adapts https://github.com/zongyi-li/fourier_neural_operator.

MIT License, Copyright (c) 2020 Zongyi Li. See THIRD_PARTY_NOTICES.md.

Both models map ``x`` of shape ``[batch, *space, initial_step * channels]`` and
a coordinate grid ``[batch, *space, dim]`` to the next state
``[batch, *space, 1, channels]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, num_channels: int, modes: int = 12, width: int = 20, initial_step: int = 1):
        super().__init__()
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = self.fc0(torch.cat((x, grid), dim=-1)).permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])
        for conv, w, act in (
            (self.conv0, self.w0, True), (self.conv1, self.w1, True),
            (self.conv2, self.w2, True), (self.conv3, self.w3, False),
        ):
            x = conv(x) + w(x)
            if act:
                x = F.gelu(x)
        x = x[..., : -self.padding].permute(0, 2, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x.unsqueeze(-2)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.out_channels = out_channels
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1)
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, num_channels: int, modes1: int = 12, modes2: int = 12,
                 width: int = 20, initial_step: int = 1):
        super().__init__()
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = self.fc0(torch.cat((x, grid), dim=-1)).permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        for conv, w, act in (
            (self.conv0, self.w0, True), (self.conv1, self.w1, True),
            (self.conv2, self.w2, True), (self.conv3, self.w3, False),
        ):
            x = conv(x) + w(x)
            if act:
                x = F.gelu(x)
        x = x[..., : -self.padding, : -self.padding].permute(0, 2, 3, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x.unsqueeze(-2)


def build_fno(spatial_dims: int, num_channels: int, modes: int, width: int,
              initial_step: int = 1) -> nn.Module:
    if spatial_dims == 1:
        return FNO1d(num_channels, modes=modes, width=width, initial_step=initial_step)
    if spatial_dims == 2:
        return FNO2d(num_channels, modes1=modes, modes2=modes, width=width, initial_step=initial_step)
    raise ValueError(f"unsupported spatial dimension: {spatial_dims}")
