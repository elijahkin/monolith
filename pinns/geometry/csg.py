"""Constructive solid geometry (CSG) operations on domains.

CSG is a technique used by 3D game engines and CAD programs for creating
complex surfaces out of simpler objects. In particular, the system relies on
three binary operations for objects (as sets):
- Union: Merge two objects into one; "gluing" in physical analogy.
- Difference: Subtraction of one object from another; "carving" in physical analogy.
- Intersection: Portion common to both objects

Of course, in order to use these operations, we must have some objects to
start with. These are so called *geometric primitives*.

https://en.wikipedia.org/wiki/Constructive_solid_geometry
"""

import torch

from .domain import Domain


class CSGDifference(Domain):
    """Constructs domain A \ B via phi_{A \ B}(x) = max(phi_A(x), -phi_B(x))."""

    def __init__(self, domain_a: Domain, domain_b: Domain):
        assert domain_a.dim == domain_b.dim
        assert (
            domain_a.low is not None and domain_a.high is not None
        ), "domain_a must have a bounding box."

        def diff_sdf(x: torch.Tensor) -> torch.Tensor:
            return torch.maximum(domain_a.compute_sdf(x), -domain_b.compute_sdf(x))

        super().__init__(
            domain_a.dim,
            diff_sdf,
            (domain_a.low.tolist(), domain_a.high.tolist()),
        )


class CSGUnion(Domain):
    """Constructs domain A U B via phi_{A U B}(x) = min(phi_A(x), phi_B(x))."""

    def __init__(self, domain_a: Domain, domain_b: Domain):
        assert domain_a.dim == domain_b.dim
        assert (
            domain_a.low is not None
            and domain_a.high is not None
            and domain_b.low is not None
            and domain_b.high is not None
        ), "Domains must have bounding boxes."
        low = [
            min(l1, l2) for l1, l2 in zip(domain_a.low.tolist(), domain_b.low.tolist())
        ]
        high = [
            max(h1, h2)
            for h1, h2 in zip(domain_a.high.tolist(), domain_b.high.tolist())
        ]

        def union_sdf(x: torch.Tensor) -> torch.Tensor:
            return torch.minimum(domain_a.compute_sdf(x), domain_b.compute_sdf(x))

        super().__init__(domain_a.dim, union_sdf, (low, high))


class CSGIntersection(Domain):
    """Constructs domain A & B via phi_{A & B}(x) = max(phi_A(x), phi_B(x))."""

    def __init__(self, domain_a: Domain, domain_b: Domain):
        assert domain_a.dim == domain_b.dim
        assert (
            domain_a.low is not None
            and domain_a.high is not None
            and domain_b.low is not None
            and domain_b.high is not None
        ), "Domains must have bounding boxes."
        low = [
            max(l1, l2) for l1, l2 in zip(domain_a.low.tolist(), domain_b.low.tolist())
        ]
        high = [
            min(h1, h2)
            for h1, h2 in zip(domain_a.high.tolist(), domain_b.high.tolist())
        ]

        def inter_sdf(x: torch.Tensor) -> torch.Tensor:
            return torch.maximum(domain_a.compute_sdf(x), domain_b.compute_sdf(x))

        super().__init__(domain_a.dim, inter_sdf, (low, high))
