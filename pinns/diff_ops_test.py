"""Tests for differential operators."""

import pytest
import torch

import diff_ops

NonlinearPoissonOperator = diff_ops.DifferentialOperator.from_callable(
    lambda u, x, k=1.0: diff_ops.laplacian(u, x)
    + k * (u**3)
    - torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])
)


def _navier_stokes_2d(u_vector, x, nu=0.01):
    u, v, p = u_vector[:, 0:1], u_vector[:, 1:2], u_vector[:, 2:3]
    du, dv, dp = diff_ops.grad(u, x), diff_ops.grad(v, x), diff_ops.grad(p, x)
    lap_u, lap_v = diff_ops.laplacian(u, x), diff_ops.laplacian(v, x)
    return {
        "momentum_u": u * du[:, 0:1] + v * du[:, 1:2] + dp[:, 0:1] - nu * lap_u,
        "momentum_v": u * dv[:, 0:1] + v * dv[:, 1:2] + dp[:, 1:2] - nu * lap_v,
        "continuity": du[:, 0:1] + dv[:, 1:2],
    }


NavierStokes2D = diff_ops.DifferentialOperator.from_callable(_navier_stokes_2d)


def _nonlinear_schrodinger(psi_vector, xt, gamma=1.0):
    u, v = psi_vector[:, 0:1], psi_vector[:, 1:2]
    spatial_axes = range(xt.shape[1] - 1)
    lap_u, lap_v = (
        diff_ops.axis_laplacian(u, xt, axes=spatial_axes),
        diff_ops.axis_laplacian(v, xt, axes=spatial_axes),
    )
    du_dt, dv_dt = diff_ops.partial(u, xt, axis=-1), diff_ops.partial(v, xt, axis=-1)
    psi_sq_mag = u**2 + v**2
    return {
        "real": -dv_dt + 0.5 * lap_u + gamma * psi_sq_mag * u,
        "imag": du_dt + 0.5 * lap_v + gamma * psi_sq_mag * v,
    }


NonlinearSchrodingerOperator = diff_ops.DifferentialOperator.from_callable(
    _nonlinear_schrodinger
)


AllenCahnOperator = diff_ops.DifferentialOperator.from_callable(
    lambda u, xt, mu=1.0: diff_ops.partial(u, xt, axis=-1)
    - mu * diff_ops.axis_laplacian(u, xt, axes=range(xt.shape[1] - 1))
    - u
    + u**3
)


def test_nonlinear_poisson_operator():
    x = torch.tensor([[1.0, 1.0]], requires_grad=True)
    u = x[:, 0:1] ** 2 + x[:, 1:2] ** 2
    res = NonlinearPoissonOperator(u, x, k=2.0)
    expected = 4.0 + 16.0 - torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])
    torch.testing.assert_close(res, expected)


def test_navier_stokes_2d_operator():
    x = torch.rand(10, 2, requires_grad=True)
    u_vec = torch.stack([x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1]], dim=1)
    residuals = NavierStokes2D(u_vec, x, nu=0.1)
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
    xt = torch.rand(10, 2, requires_grad=True)
    psi_vector = torch.stack([xt[:, 0] ** 2, xt[:, 1] ** 2], dim=1)
    residuals = NonlinearSchrodingerOperator(psi_vector, xt, gamma=2.0)
    assert "real" in residuals
    assert "imag" in residuals
    assert residuals["real"].shape == (10, 1)
    assert residuals["imag"].shape == (10, 1)

    xt_fixed = torch.tensor([[1.0, 1.0]], requires_grad=True)
    psi_fixed = torch.stack([xt_fixed[:, 0] ** 2, xt_fixed[:, 1] ** 2], dim=1)
    res_fixed = NonlinearSchrodingerOperator(psi_fixed, xt_fixed, gamma=2.0)
    torch.testing.assert_close(res_fixed["real"], torch.tensor([[3.0]]))
    torch.testing.assert_close(res_fixed["imag"], torch.tensor([[4.0]]))


def test_allen_cahn_operator():
    xt = torch.rand(10, 2, requires_grad=True)
    u = xt[:, 0:1] ** 2 + xt[:, 1:2] ** 2
    res = AllenCahnOperator(u, xt, mu=2.0)
    assert res.shape == (10, 1)

    xt_fixed = torch.tensor([[1.0, 1.0]], requires_grad=True)
    u_fixed = xt_fixed[:, 0:1] ** 2 + xt_fixed[:, 1:2] ** 2
    res_fixed = AllenCahnOperator(u_fixed, xt_fixed, mu=2.0)
    torch.testing.assert_close(res_fixed, torch.tensor([[4.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
