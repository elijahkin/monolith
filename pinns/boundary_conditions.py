"""Boundary and initial condition specifications for BVPs."""

from typing import Callable, List, Optional, Union, Tuple
from diff_ops import grad
import torch


class BoundaryCondition:
    """Base class for boundary conditions applying on all or part of Gamma."""

    def __init__(
        self,
        on_boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        component: Optional[Union[int, List[int]]] = None,
    ):
        self.on_boundary_fn = (
            on_boundary_fn
            if on_boundary_fn
            else (lambda x: torch.ones(x.shape[0], dtype=torch.bool, device=x.device))
        )
        self.component = component

    def filter_boundary(
        self, x: torch.Tensor, normal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.on_boundary_fn(x)
        return x[mask], normal[mask]

    def residual(
        self, model: Callable, x: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class DirichletBC(BoundaryCondition):
    """u(x) = g(x) on Gamma."""

    def __init__(
        self,
        func_or_value: Union[
            float, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
        ],
        on_boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        component: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(on_boundary_fn, component)
        self.value = func_or_value

    def residual(
        self, model: Callable, x: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        x_sub, _ = self.filter_boundary(x, normal)
        if x_sub.shape[0] == 0:
            return torch.empty(0, 1, device=x.device)

        u = model(x_sub)
        if self.component is not None:
            u = u[
                :,
                (
                    self.component
                    if isinstance(self.component, list)
                    else [self.component]
                ),
            ]

        target = self.value(x_sub) if callable(self.value) else self.value
        if not isinstance(target, torch.Tensor):
            target = torch.full_like(u, target)
        return u - target


class NeumannBC(BoundaryCondition):
    """du/dn = grad(u) . n = h(x) on Gamma."""

    def __init__(
        self,
        func_or_value: Union[
            float, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
        ],
        on_boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        component: int = 0,
    ):
        super().__init__(on_boundary_fn, component=component)
        self.value = func_or_value

    def residual(
        self, model: Callable, x: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        x_sub, n_sub = self.filter_boundary(x, normal)
        if x_sub.shape[0] == 0:
            return torch.empty(0, 1, device=x.device)

        x_sub = x_sub.clone().detach().requires_grad_(True)
        u = model(x_sub)[:, self.component : self.component + 1]

        du_dx = grad(u, x_sub)
        normal_deriv = (du_dx * n_sub).sum(dim=-1, keepdim=True)

        target = self.value(x_sub) if callable(self.value) else self.value
        if not isinstance(target, torch.Tensor):
            target = torch.full_like(normal_deriv, target)
        return normal_deriv - target


class PeriodicBC(BoundaryCondition):
    """Enforces u(x_src) == u(x_dst) (and optionally derivatives) across periodic boundary pairs."""

    def __init__(
        self,
        on_source_boundary_fn: Callable[[torch.Tensor], torch.Tensor],
        map_to_dst_fn: Callable[[torch.Tensor], torch.Tensor],
        component: Optional[Union[int, List[int]]] = None,
        enforce_derivative: bool = False,
    ):
        super().__init__(on_source_boundary_fn, component)
        self.map_fn = map_to_dst_fn
        self.enforce_derivative = enforce_derivative

    def residual(
        self, model: Callable, x: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        x_src, _ = self.filter_boundary(x, normal)
        if x_src.shape[0] == 0:
            return torch.empty(0, 1, device=x.device)

        x_dst = self.map_fn(x_src)

        if not self.enforce_derivative:
            u_src = model(x_src)
            u_dst = model(x_dst)
            if self.component is not None:
                u_src = u_src[
                    :,
                    (
                        self.component
                        if isinstance(self.component, list)
                        else [self.component]
                    ),
                ]
                u_dst = u_dst[
                    :,
                    (
                        self.component
                        if isinstance(self.component, list)
                        else [self.component]
                    ),
                ]
            return u_src - u_dst
        else:
            x_src = x_src.clone().detach().requires_grad_(True)
            x_dst = x_dst.clone().detach().requires_grad_(True)
            u_src = model(x_src)[:, self.component : self.component + 1]
            u_dst = model(x_dst)[:, self.component : self.component + 1]
            return grad(u_src, x_src) - grad(u_dst, x_dst)


class InitialCondition:
    """Enforces initial condition u(x, t0) = u0(x) or time derivatives."""

    def __init__(
        self,
        u0_fn: Union[float, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
        time_index: int = -1,
        t0: float = 0.0,
        derivative_order: int = 0,
        component: Optional[Union[int, List[int]]] = None,
    ):
        self.u0_fn = u0_fn
        self.time_index = time_index
        self.t0 = t0
        self.derivative_order = derivative_order
        self.component = component

    def residual(self, model: Callable, x_spatial: torch.Tensor) -> torch.Tensor:
        N = x_spatial.shape[0]
        t_col = torch.full(
            (N, 1), self.t0, dtype=x_spatial.dtype, device=x_spatial.device
        )
        xt = torch.cat([x_spatial, t_col], dim=-1)

        if self.derivative_order == 0:
            u = model(xt)
            if self.component is not None:
                u = u[
                    :,
                    (
                        self.component
                        if isinstance(self.component, list)
                        else [self.component]
                    ),
                ]
        elif self.derivative_order == 1:
            xt = xt.clone().detach().requires_grad_(True)
            u_full = model(xt)
            idx = self.component if self.component is not None else 0
            du_dxt = grad(u_full[:, idx : idx + 1], xt)
            u = du_dxt[:, self.time_index : self.time_index + 1]
        else:
            raise ValueError(f"Unsupported derivative order: {self.derivative_order}")

        target = self.u0_fn(x_spatial) if callable(self.u0_fn) else self.u0_fn
        if not isinstance(target, torch.Tensor):
            target = torch.full_like(u, target)
        return u - target
