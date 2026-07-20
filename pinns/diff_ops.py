"""Differential calculus utilities and operator specification for BVPs."""

from typing import Callable, Dict, Union
import torch


def grad(u: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Computes the gradient du_i/dx_j for scalar or component tensor u with respect to x."""
    if u.ndim == 1 or (u.ndim == 2 and u.shape[1] == 1):
        if not u.requires_grad:
            return torch.zeros_like(x)
        g = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return g if g is not None else torch.zeros_like(x)
    else:
        if not u.requires_grad:
            return torch.zeros(
                (u.shape[0], u.shape[1], x.shape[1]), device=x.device, dtype=x.dtype
            )
        jac = []
        for i in range(u.shape[1]):
            g = torch.autograd.grad(
                u[:, i],
                x,
                grad_outputs=torch.ones_like(u[:, i]),
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
            jac.append(g if g is not None else torch.zeros_like(x))
        return torch.stack(jac, dim=1)


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes div(grad(u)) for a scalar field u."""
    du_dx = grad(u, x, create_graph=True)
    lap = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for j in range(x.shape[1]):
        if not du_dx[:, j].requires_grad:
            continue
        g = torch.autograd.grad(
            du_dx[:, j],
            x,
            grad_outputs=torch.ones_like(du_dx[:, j]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if g is not None:
            lap = lap + g[:, j]
    return lap.unsqueeze(-1)


def divergence(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes divergence of a vector field u (where dim d_u == d_x)."""
    div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(min(u.shape[1], x.shape[1])):
        if not u[:, i].requires_grad:
            continue
        g = torch.autograd.grad(
            u[:, i],
            x,
            grad_outputs=torch.ones_like(u[:, i]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if g is not None:
            div = div + g[:, i]
    return div.unsqueeze(-1)


class DifferentialOperator:
    """Base class for defining differential operators L(u, x) = 0."""

    def __call__(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        **params: Union[float, torch.Tensor],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError
