"""Tests for boundary conditions."""

from boundary_conditions import DirichletBC, NeumannBC, PeriodicBC, InitialCondition
import pytest
import torch


def test_dirichlet_bc():
    bc = DirichletBC(func_or_value=5.0, component=0)
    model = lambda x: torch.zeros(x.shape[0], 1)
    x_b = torch.rand(10, 2)
    n_b = torch.rand(10, 2)
    res = bc.residual(model, x_b, n_b)
    torch.testing.assert_close(res, torch.full((10, 1), -5.0))


def test_neumann_bc():
    bc = NeumannBC(func_or_value=2.0, component=0)
    model = lambda x: 3.0 * x[:, 0:1] + 1.0 * x[:, 1:2]
    x_b = torch.rand(5, 2)
    n_b = torch.tensor([[1.0, 0.0]] * 5)
    res = bc.residual(model, x_b, n_b)
    torch.testing.assert_close(res, torch.full((5, 1), 1.0))


def test_periodic_bc():
    bc = PeriodicBC(
        on_source_boundary_fn=lambda x: x[:, 0] < 0,
        map_to_dst_fn=lambda x: torch.stack([x[:, 0] + 2.0, x[:, 1]], dim=-1),
        component=0,
    )
    model = lambda x: x[:, 0:1]
    x_src = torch.tensor([[-1.0, 0.5]])
    n_src = torch.tensor([[-1.0, 0.0]])
    res = bc.residual(model, x_src, n_src)
    torch.testing.assert_close(res, torch.tensor([[-2.0]]))


def test_initial_condition():
    ic_val = InitialCondition(
        u0_fn=lambda x: torch.sin(x[:, 0:1]),
        time_index=1,
        t0=0.0,
        derivative_order=0,
    )
    model = lambda xt: xt[:, 0:1] + xt[:, 1:2]
    x_spatial = torch.tensor([[torch.pi / 2]])
    res = ic_val.residual(model, x_spatial)
    expected = torch.tensor([[torch.pi / 2 - 1.0]])
    torch.testing.assert_close(res, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
