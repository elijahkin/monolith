"""Tests for geometry.py."""

from geometry.primitives import Ball, Interval
import pytest
import torch


def test_interval_and_cartesian_product_rectangle():
    int_x = Interval(-1.0, 1.0)
    int_y = Interval(0.0, 2.0)
    rect = int_x * int_y

    assert rect.dim == 2
    assert torch.all(rect.is_inside(torch.tensor([[0.0, 1.0]]))).item()
    assert not torch.any(rect.is_inside(torch.tensor([[2.0, 1.0]]))).item()

    x_b, n_b = rect.sample_boundary(200)
    assert x_b.shape == (200, 2)
    assert n_b.shape == (200, 2)
    norms = torch.linalg.norm(n_b, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_ball_and_cartesian_product_cylinder():
    int_z = Interval(0.0, 5.0)
    disk_xy = Ball(dim=2, radius=1.0, center=[0.0, 0.0])
    cylinder = int_z * disk_xy

    assert cylinder.dim == 3
    assert torch.all(cylinder.is_inside(torch.tensor([[2.5, 0.5, 0.5]]))).item()
    assert not torch.any(cylinder.is_inside(torch.tensor([[2.5, 1.5, 0.0]]))).item()

    x_b, n_b = cylinder.sample_boundary(100)
    assert x_b.shape[1] == 3
    assert n_b.shape[1] == 3


def test_csg_operations():
    box = Interval(-2.0, 2.0) * Interval(-2.0, 2.0)
    disk = Ball(dim=2, radius=1.0)
    diff = box - disk

    assert not torch.any(diff.is_inside(torch.tensor([[0.0, 0.0]]))).item()
    assert torch.all(diff.is_inside(torch.tensor([[1.5, 1.5]]))).item()


def test_compute_normal_via_autodiff():
    disk = Ball(dim=2, radius=1.0)
    x_pt = torch.tensor([[1.0, 0.0]])
    normal = disk.compute_normal(x_pt)
    torch.testing.assert_close(normal, torch.tensor([[1.0, 0.0]]), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
