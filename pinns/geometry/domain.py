"""Domain base class and Cartesian product composition."""

from typing import Callable, Optional, Sequence, Tuple
import torch


class Domain:
    """Unified base class for spatial and spatio-temporal domains backed by Signed Distance Functions (SDFs)."""

    def __init__(
        self,
        dim: int,
        sdf_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        bounding_box: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    ):
        self.dim = dim
        self._sdf_fn = sdf_fn
        self.low: Optional[torch.Tensor] = (
            torch.tensor(bounding_box[0], dtype=torch.float32)
            if bounding_box is not None
            else None
        )
        self.high: Optional[torch.Tensor] = (
            torch.tensor(bounding_box[1], dtype=torch.float32)
            if bounding_box is not None
            else None
        )

    def compute_sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the signed distance function phi(x). (<0 inside, ==0 boundary, >0 outside)."""
        if self._sdf_fn is not None:
            return self._sdf_fn(x)
        raise NotImplementedError(
            "Domain subclass must provide sdf_fn in __init__ or override"
            " compute_sdf()."
        )

    def is_inside(self, x: torch.Tensor) -> torch.Tensor:
        """Checks if x is inside the domain (phi(x) < 0)."""
        return self.compute_sdf(x).squeeze(-1) < 0.0

    def compute_normal(self, x: torch.Tensor) -> torch.Tensor:
        """Computes unit outward normal n = grad(phi) / ||grad(phi)|| via automatic differentiation."""
        x_req = x.clone().detach().requires_grad_(True)
        phi = self.compute_sdf(x_req)
        dphi = torch.autograd.grad(
            phi, x_req, grad_outputs=torch.ones_like(phi), create_graph=True
        )[0]
        norm = torch.linalg.norm(dphi, dim=-1, keepdim=True) + 1e-12
        return dphi / norm

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Samples n interior points. Defaults to rejection sampling within bounding_box."""
        assert (
            self.low is not None and self.high is not None
        ), "bounding_box required for default interior sampling."
        points = []
        low = self.low.to(device)
        high = self.high.to(device)
        while sum(p.shape[0] for p in points) < n:
            cand = low + (high - low) * torch.rand(n, self.dim, device=device)
            mask = self.is_inside(cand)
            points.append(cand[mask])
        return torch.cat(points, dim=0)[:n]

    def sample_boundary(
        self, n: int, device: str = "cpu", tol: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples n boundary points and normals. Defaults to rejection sampling near phi(x) == 0."""
        assert (
            self.low is not None and self.high is not None
        ), "bounding_box required for default boundary sampling."
        points = []
        low = self.low.to(device)
        high = self.high.to(device)
        while sum(p.shape[0] for p in points) < n:
            cand = low + (high - low) * torch.rand(n * 10, self.dim, device=device)
            phi = torch.abs(self.compute_sdf(cand).squeeze(-1))
            mask = phi < tol
            points.append(cand[mask])
        x_b = torch.cat(points, dim=0)[:n]
        n_b = self.compute_normal(x_b)
        return x_b, n_b

    def __mul__(self, other: "Domain") -> "CartesianProduct":
        return CartesianProduct(self, other)

    def __sub__(self, other: "Domain") -> "CSGDifference":
        from .csg import CSGDifference

        return CSGDifference(self, other)

    def __or__(self, other: "Domain") -> "CSGUnion":
        from .csg import CSGUnion

        return CSGUnion(self, other)

    def __and__(self, other: "Domain") -> "CSGIntersection":
        from .csg import CSGIntersection

        return CSGIntersection(self, other)


class CartesianProduct(Domain):
    """Cartesian product C = A * B in R^{d_A + d_B} with exact composite SDF and stratified boundary sampling."""

    def __init__(self, domain_a: Domain, domain_b: Domain):
        assert (
            domain_a.low is not None
            and domain_a.high is not None
            and domain_b.low is not None
            and domain_b.high is not None
        ), "Domains must have bounding boxes."
        dim = domain_a.dim + domain_b.dim
        low = domain_a.low.tolist() + domain_b.low.tolist()
        high = domain_a.high.tolist() + domain_b.high.tolist()

        def product_sdf(x: torch.Tensor) -> torch.Tensor:
            x_a = x[:, : domain_a.dim]
            x_b = x[:, domain_a.dim :]
            phi_a = domain_a.compute_sdf(x_a)
            phi_b = domain_b.compute_sdf(x_b)
            u = torch.cat([phi_a, phi_b], dim=-1)
            outside = torch.linalg.norm(
                torch.maximum(u, torch.zeros_like(u)), dim=-1, keepdim=True
            )
            inside = torch.minimum(
                torch.max(u, dim=-1, keepdim=True)[0], torch.zeros_like(outside)
            )
            return outside + inside

        super().__init__(dim=dim, sdf_fn=product_sdf, bounding_box=(low, high))
        self.domain_a = domain_a
        self.domain_b = domain_b

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Exact compositional interior sampling: A^circ x B^circ."""
        pts_a = self.domain_a.sample_interior(n, device=device)
        pts_b = self.domain_b.sample_interior(n, device=device)
        return torch.cat([pts_a, pts_b], dim=-1)

    def sample_boundary(
        self, n: int, device: str = "cpu", tol: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact product boundary formula: partial(A x B) = (partial A x A^circ) U (A^circ x partial B)."""
        n_a_bound = n // 2
        n_b_bound = n - n_a_bound

        x_ba, n_ba = self.domain_a.sample_boundary(n_a_bound, device=device, tol=tol)
        x_ib = self.domain_b.sample_interior(n_a_bound, device=device)
        n_ib = torch.zeros_like(x_ib)
        x_part1 = torch.cat([x_ba, x_ib], dim=-1)
        n_part1 = torch.cat([n_ba, n_ib], dim=-1)

        x_ia = self.domain_a.sample_interior(n_b_bound, device=device)
        n_ia = torch.zeros_like(x_ia)
        x_bb, n_bb = self.domain_b.sample_boundary(n_b_bound, device=device, tol=tol)
        x_part2 = torch.cat([x_ia, x_bb], dim=-1)
        n_part2 = torch.cat([n_ia, n_bb], dim=-1)

        return torch.cat([x_part1, x_part2], dim=0), torch.cat(
            [n_part1, n_part2], dim=0
        )
