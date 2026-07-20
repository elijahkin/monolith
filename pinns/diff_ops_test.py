"""Tests for differential operators."""

from typing import Dict

import pytest
import torch

import diff_ops


class NonlinearPoissonOperator(diff_ops.DifferentialOperator):
    """Example scalar PDE: nabla^2 u + k * u^3 - f(x) = 0."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def __call__(self, u: torch.Tensor, x: torch.Tensor, **params) -> torch.Tensor:
        lap_u = diff_ops.laplacian(u, x)
        source = torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])
        return lap_u + self.k * (u**3) - source


class NavierStokes2D(diff_ops.DifferentialOperator):
    """Example vector PDE: 2D Incompressible Navier-Stokes."""

    def __init__(self, nu: float = 0.01):
        self.nu = nu

    def __call__(
        self, u_vector: torch.Tensor, x: torch.Tensor, **params
    ) -> Dict[str, torch.Tensor]:
        u = u_vector[:, 0:1]
        v = u_vector[:, 1:2]
        p = u_vector[:, 2:3]

        du = diff_ops.grad(u, x)
        dv = diff_ops.grad(v, x)
        dp = diff_ops.grad(p, x)

        lap_u = diff_ops.laplacian(u, x)
        lap_v = diff_ops.laplacian(v, x)

        momentum_u = u * du[:, 0:1] + v * du[:, 1:2] + dp[:, 0:1] - self.nu * lap_u
        momentum_v = u * dv[:, 0:1] + v * dv[:, 1:2] + dp[:, 1:2] - self.nu * lap_v
        continuity = du[:, 0:1] + dv[:, 1:2]

        return {
            "momentum_u": momentum_u,
            "momentum_v": momentum_v,
            "continuity": continuity,
        }


class NonlinearSchrodingerOperator(diff_ops.DifferentialOperator):
    """Time-dependent Nonlinear Schrodinger Equation (NLSE) operator for psi = u + i*v.

    i * d(psi)/dt + 0.5 * laplacian(psi) + gamma * |psi|^2 * psi = 0.
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(
        self, psi_vector: torch.Tensor, xt: torch.Tensor, **params
    ) -> Dict[str, torch.Tensor]:
        u = psi_vector[:, 0:1]
        v = psi_vector[:, 1:2]

        du_dxt = diff_ops.grad(u, xt, create_graph=True)
        dv_dxt = diff_ops.grad(v, xt, create_graph=True)
        du_dt = du_dxt[:, -1:]
        dv_dt = dv_dxt[:, -1:]

        lap_u = torch.zeros(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)
        lap_v = torch.zeros(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)
        for j in range(xt.shape[1] - 1):
            d2u = torch.autograd.grad(
                du_dxt[:, j],
                xt,
                grad_outputs=torch.ones_like(du_dxt[:, j]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if d2u is not None:
                lap_u = lap_u + d2u[:, j : j + 1]

            d2v = torch.autograd.grad(
                dv_dxt[:, j],
                xt,
                grad_outputs=torch.ones_like(dv_dxt[:, j]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if d2v is not None:
                lap_v = lap_v + d2v[:, j : j + 1]

        psi_sq_mag = u**2 + v**2

        res_real = -dv_dt + 0.5 * lap_u + self.gamma * psi_sq_mag * u
        res_imag = du_dt + 0.5 * lap_v + self.gamma * psi_sq_mag * v

        return {
            "real": res_real,
            "imag": res_imag,
        }


def test_nonlinear_poisson_operator():
    op = NonlinearPoissonOperator(k=2.0)
    x = torch.tensor([[1.0, 1.0]], requires_grad=True)
    u = x[:, 0:1] ** 2 + x[:, 1:2] ** 2
    res = op(u, x)
    expected = 4.0 + 16.0 - torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])
    torch.testing.assert_close(res, expected)


def test_navier_stokes_2d_operator():
    op = NavierStokes2D(nu=0.1)
    x = torch.rand(10, 2, requires_grad=True)
    u_vec = torch.stack([x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1]], dim=1)
    residuals = op(u_vec, x)
    assert "momentum_u" in residuals
    assert "momentum_v" in residuals
    assert "continuity" in residuals
    assert residuals["continuity"].shape == (10, 1)


def test_divergence():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = torch.stack([3.0 * x[:, 0] ** 2, 4.0 * x[:, 1] ** 2], dim=1)
    div = diff_ops.divergence(u, x)
    torch.testing.assert_close(div, torch.tensor([[22.0]]))


def test_nonlinear_schrodinger_operator():
    op = NonlinearSchrodingerOperator(gamma=2.0)
    xt = torch.rand(10, 2, requires_grad=True)
    psi_vector = torch.stack([xt[:, 0] ** 2, xt[:, 1] ** 2], dim=1)
    residuals = op(psi_vector, xt)
    assert "real" in residuals
    assert "imag" in residuals
    assert residuals["real"].shape == (10, 1)
    assert residuals["imag"].shape == (10, 1)

    xt_fixed = torch.tensor([[1.0, 1.0]], requires_grad=True)
    psi_fixed = torch.stack([xt_fixed[:, 0] ** 2, xt_fixed[:, 1] ** 2], dim=1)
    res_fixed = op(psi_fixed, xt_fixed)
    torch.testing.assert_close(res_fixed["real"], torch.tensor([[3.0]]))
    torch.testing.assert_close(res_fixed["imag"], torch.tensor([[4.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
