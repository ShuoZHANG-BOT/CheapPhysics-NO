"""Physics-informed (label-free) pretraining losses.

Every loss takes a batch of cheap inputs ``a`` with shape ``[batch, *space, channel]``
and never sees a solution label. Time-dependent problems are rolled out
autoregressively from the initial condition, exactly as in supervised training,
and the governing-equation residual is evaluated on the predicted trajectory.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def rollout(model: nn.Module, initial: torch.Tensor, grid: torch.Tensor, steps: int) -> torch.Tensor:
    """Autoregressive rollout. ``initial``: ``[batch, x, 1, channel]``; returns ``[batch, x, steps, channel]``."""
    window, predictions = initial, [initial]
    shape = [initial.shape[0], initial.shape[1], -1]
    for _ in range(initial.shape[-2], steps):
        prediction = model(window.reshape(shape), grid)
        predictions.append(prediction)
        window = torch.cat((window[..., 1:, :], prediction), dim=-2)
    return torch.cat(predictions, dim=-2)


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def burgers_residual(u: torch.Tensor, x: np.ndarray, t: np.ndarray, viscosity: float) -> torch.Tensor:
    """u_t + u u_x - nu u_xx on a periodic grid; central differences in time, spectral in space.

    ``u``: ``[batch, x, time]``. Returns the residual at interior times ``[batch, x, time-2]``.
    """
    t = t[: u.shape[-1]]
    dt = torch.as_tensor(t[2:] - t[:-2], device=u.device, dtype=u.dtype).view(1, 1, -1)
    u_t = (u[:, :, 2:] - u[:, :, :-2]) / dt
    middle = u[:, :, 1:-1]
    dx = float(np.mean(np.diff(x)))
    k = (2.0 * math.pi * torch.fft.fftfreq(middle.shape[1], d=dx, device=u.device)).to(u.dtype).view(1, -1, 1)
    spectrum = torch.fft.fft(middle, dim=1)
    u_x = torch.fft.ifft(1j * k * spectrum, dim=1).real
    u_xx = torch.fft.ifft(-k.square() * spectrum, dim=1).real
    return u_t + middle * u_x - viscosity * u_xx


def advection_residual(u: torch.Tensor, dx: float, dt: float, beta: float) -> torch.Tensor:
    """u_t + beta u_x; forward difference in time and centered periodic difference in space."""
    now, previous = u[:, :, 1:], u[:, :, :-1]
    u_x = (torch.roll(now, -1, dims=1) - torch.roll(now, 1, dims=1)) / (2.0 * dx)
    return (now - previous) / dt + beta * u_x


def euler_conserved_and_flux(state: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Channels ordered (density, pressure, Vx) as in PDEBench."""
    density, pressure, velocity = state.unbind(dim=-1)
    momentum = density * velocity
    energy = pressure / (gamma - 1.0) + 0.5 * density * velocity.square()
    conserved = torch.stack((density, momentum, energy), dim=-1)
    flux = torch.stack((momentum, momentum * velocity + pressure, velocity * (energy + pressure)), dim=-1)
    return conserved, flux


def euler_residual(prediction: torch.Tensor, dx: float, dt: float, gamma: float) -> torch.Tensor:
    """Inviscid Euler conservation q^{n+1} - q^n + dt d_x F(q^{n+1}) on a periodic grid.

    PDEBench 1D CFD data contain small viscosity, so this residual is an approximate physics model.
    """
    previous, current = prediction[:, :, :-1], prediction[:, :, 1:]
    q_previous, _ = euler_conserved_and_flux(previous, gamma)
    q_current, flux = euler_conserved_and_flux(current, gamma)
    flux_x = (torch.roll(flux, -1, dims=1) - torch.roll(flux, 1, dims=1)) / (2.0 * dx)
    return q_current - q_previous + dt * flux_x


def darcy_residual(coef: torch.Tensor, u: torch.Tensor, dx: float, dy: float, beta: float) -> torch.Tensor:
    """-div(a grad u) - beta at interior nodes, finite-volume face averages of ``a``."""
    c = coef[:, 1:-1, 1:-1]
    axp, axm = 0.5 * (coef[:, 2:, 1:-1] + c), 0.5 * (coef[:, :-2, 1:-1] + c)
    ayp, aym = 0.5 * (coef[:, 1:-1, 2:] + c), 0.5 * (coef[:, 1:-1, :-2] + c)
    uc = u[:, 1:-1, 1:-1]
    div = ((axp * (u[:, 2:, 1:-1] - uc) - axm * (uc - u[:, :-2, 1:-1])) / dx ** 2
           + (ayp * (u[:, 1:-1, 2:] - uc) - aym * (uc - u[:, 1:-1, :-2])) / dy ** 2)
    return -div - beta


# ---------------------------------------------------------------------------
# Per-dataset pretraining objectives
# ---------------------------------------------------------------------------

class PhysicsLoss:
    """Base class. ``__call__(model, a)`` returns the scalar PI loss for a batch of cheap inputs."""

    def __init__(self, cfg: dict, grid, cheap_inputs: torch.Tensor, device: torch.device):
        self.cfg, self.device = cfg, device
        self.weights = cfg["pi"].get("weights", {})
        self.coords = grid.coords
        self.attrs = grid.attrs
        self.grid = grid.grid.to(device)
        self.steps = cfg["data"]["t_train"]

    def batch_grid(self, batch: int) -> torch.Tensor:
        return self.grid.unsqueeze(0).expand(batch, *self.grid.shape)

    def __call__(self, model: nn.Module, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BurgersLoss(PhysicsLoss):
    """PDE residual + initial condition + periodic boundary penalty (Burgers, nu = Nu / pi)."""

    def __init__(self, cfg, grid, cheap_inputs, device):
        super().__init__(cfg, grid, cheap_inputs, device)
        # PDEBench's Burgers generator uses Nu / pi as the diffusion coefficient.
        self.viscosity = float(self.attrs.get("Nu", 0.01)) / math.pi

    def __call__(self, model, a):
        prediction = rollout(model, a.unsqueeze(-2), self.batch_grid(len(a)), self.steps)
        residual = burgers_residual(prediction[..., 0], self.coords["x"], self.coords["t"], self.viscosity)
        ic = (prediction[..., 0, 0] - a[..., 0]).square().mean()
        bc = (prediction[:, 0] - prediction[:, -1]).square().mean()
        w = self.weights
        return w.get("pde", 1.0) * residual.square().mean() + w.get("ic", 1.0) * ic + w.get("bc", 0.1) * bc


class AdvectionLoss(PhysicsLoss):
    def __init__(self, cfg, grid, cheap_inputs, device):
        super().__init__(cfg, grid, cheap_inputs, device)
        x, t = self.coords["x"], self.coords["t"]
        self.dx, self.dt = float(x[1] - x[0]), float(t[1] - t[0])
        self.beta = float(self.attrs["beta"])
        self.steps = min(self.steps, len(t))

    def __call__(self, model, a):
        prediction = rollout(model, a.unsqueeze(-2), self.batch_grid(len(a)), self.steps)
        residual = advection_residual(prediction[..., 0], self.dx, self.dt, self.beta)
        # Scaling by dt keeps the minimizer and stabilizes the loss magnitude.
        pde = (self.dt * residual).square().mean()
        bc = (prediction[:, 0] - prediction[:, -1]).square().mean()
        return self.weights.get("pde", 1.0) * pde + self.weights.get("bc", 0.1) * bc


class EulerLoss(PhysicsLoss):
    """Approximate (inviscid) physics for PDEBench 1D CFD, normalized per conserved variable."""

    def __init__(self, cfg, grid, cheap_inputs, device):
        super().__init__(cfg, grid, cheap_inputs, device)
        x, t = self.coords["x"], self.coords["t"]
        self.dx, self.dt = float(x[1] - x[0]), float(t[1] - t[0])
        self.gamma = float(cfg["pi"].get("gamma", 5.0 / 3.0))
        self.steps = min(self.steps, len(t))
        q, _ = euler_conserved_and_flux(cheap_inputs, self.gamma)
        self.q_scale = q.std(dim=(0, 1), unbiased=False).clamp_min(1e-3).to(device).view(1, 1, 1, 3)

    def __call__(self, model, a):
        prediction = rollout(model, a.unsqueeze(-2), self.batch_grid(len(a)), self.steps)
        residual = euler_residual(prediction, self.dx, self.dt, self.gamma)
        pde = (residual / self.q_scale).square().mean()
        state_scale = a.std(dim=(0, 1), unbiased=False).clamp_min(1e-3)
        bc = ((prediction[:, 0] - prediction[:, -1]) / state_scale).square().mean()
        return self.weights.get("pde", 1.0) * pde + self.weights.get("bc", 0.1) * bc


class DarcyLoss(PhysicsLoss):
    """-div(a grad u) = beta with homogeneous Dirichlet boundary; input is the coefficient a."""

    def __init__(self, cfg, grid, cheap_inputs, device):
        super().__init__(cfg, grid, cheap_inputs, device)
        x, y = self.coords["x"], self.coords["y"]
        self.dx, self.dy = float(x[1] - x[0]), float(y[1] - y[0])
        self.beta = float(self.attrs.get("beta", 1.0))

    def __call__(self, model, a):
        u = model(a, self.batch_grid(len(a)))[..., 0, 0]
        coef = a[..., 0]
        pde = darcy_residual(coef, u, self.dx, self.dy, self.beta).square().mean()
        boundary = torch.cat((u[:, 0], u[:, -1], u[:, 1:-1, 0], u[:, 1:-1, -1]), dim=1).square().mean()
        return self.weights.get("pde", 1.0) * pde + self.weights.get("bc", 10.0) * boundary


LOSSES = {"burgers": BurgersLoss, "advection": AdvectionLoss, "euler": EulerLoss, "darcy": DarcyLoss}


def build_physics_loss(cfg: dict, grid, cheap_inputs: torch.Tensor, device: torch.device) -> PhysicsLoss:
    return LOSSES[cfg["pi"]["residual"]](cfg, grid, cheap_inputs, device)
