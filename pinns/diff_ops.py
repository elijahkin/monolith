"""Differential calculus utilities and operator specification for BVPs."""

from typing import Callable, Dict, Sequence, Union
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


def partial(u: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
    """Computes du/dx_{axis} for a scalar field u along a single coordinate axis."""
    if not u.requires_grad:
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    g = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    axis = axis % x.shape[1]
    return g[:, axis : axis + 1]


def second_partial(u: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
    """Computes d^2u/dx_{axis}^2 for a scalar field u along a single coordinate axis."""
    axis = axis % x.shape[1]
    du_daxis = partial(u, x, axis)
    if not du_daxis.requires_grad:
        return torch.zeros_like(du_daxis)
    g = torch.autograd.grad(
        du_daxis,
        x,
        grad_outputs=torch.ones_like(du_daxis),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(du_daxis)
    return g[:, axis : axis + 1]


def axis_laplacian(
    u: torch.Tensor, x: torch.Tensor, axes: Sequence[int]
) -> torch.Tensor:
    """Sum of second partials of u along the given axes (div in axis subset)."""
    lap = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for j in axes:
        lap = lap + second_partial(u, x, j)
    return lap


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes div(grad(u)) for a scalar field u."""
    return axis_laplacian(u, x, range(x.shape[1]))


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
        fn = getattr(self, "_fn", None)
        if fn is not None:
            return fn(u, x, **params)
        raise NotImplementedError

    @classmethod
    def from_callable(
        cls,
        fn: Callable[
            [torch.Tensor, torch.Tensor], Union[torch.Tensor, Dict[str, torch.Tensor]]
        ],
    ) -> "DifferentialOperator":
        """Wraps fn(u, x, **params) -> Tensor | Dict[str, Tensor] as a DifferentialOperator."""
        op = cls.__new__(cls)
        op._fn = fn  # type: ignore[attr-defined]
        return op
