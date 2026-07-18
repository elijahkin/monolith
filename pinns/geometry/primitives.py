"""Geometric primitives: Interval and Ball."""

from typing import Optional, Sequence, Tuple
import torch

from .domain import Domain


class Interval(Domain):
    """1D Interval [low, high] in R^1."""

    def __init__(self, low: float, high: float):
        assert low < high, "low must be strictly less than high"
        center = (low + high) / 2.0
        half_length = (high - low) / 2.0

        def interval_sdf(x: torch.Tensor) -> torch.Tensor:
            return torch.abs(x - center) - half_length

        super().__init__(dim=1, sdf_fn=interval_sdf, bounding_box=([low], [high]))

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        assert self.low is not None and self.high is not None
        low = self.low.to(device)
        high = self.high.to(device)
        return low + (high - low) * torch.rand(n, 1, device=device)

    def sample_boundary(
        self, n: int, device: str = "cpu", tol: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Boundary of 1D interval is exactly the two endpoints {low, high} with normals {-1, +1}."""
        assert self.low is not None and self.high is not None
        n_low = n // 2
        n_high = n - n_low
        x_low = torch.full((n_low, 1), self.low.item(), device=device)
        x_high = torch.full((n_high, 1), self.high.item(), device=device)

        norm_low = torch.full((n_low, 1), -1.0, device=device)
        norm_high = torch.full((n_high, 1), 1.0, device=device)

        return torch.cat([x_low, x_high], dim=0), torch.cat(
            [norm_low, norm_high], dim=0
        )


class Ball(Domain):
    """d-dimensional Ball/Disk/Sphere {x in R^d : ||x - center||_2 <= radius}."""

    def __init__(
        self, dim: int, radius: float, center: Optional[Sequence[float]] = None
    ):
        c = center if center is not None else [0.0] * dim
        assert len(c) == dim
        center_t = torch.tensor(c, dtype=torch.float32)

        def ball_sdf(x: torch.Tensor) -> torch.Tensor:
            return (
                torch.linalg.norm(x - center_t.to(x.device), dim=-1, keepdim=True)
                - radius
            )

        low = [c_i - radius for c_i in c]
        high = [c_i + radius for c_i in c]
        super().__init__(dim=dim, sdf_fn=ball_sdf, bounding_box=(low, high))
        self.radius = radius
        self.center = center_t

    def sample_boundary(
        self, n: int, device: str = "cpu", tol: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact uniform sphere boundary sampling via Gaussian normalization."""
        gauss = torch.randn(n, self.dim, device=device)
        normals = gauss / (torch.linalg.norm(gauss, dim=-1, keepdim=True) + 1e-12)
        pts = self.center.to(device) + self.radius * normals
        return pts, normals
