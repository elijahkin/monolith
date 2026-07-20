# Physics-Informed Neural Networks

A recent method of numerically solving partial differential equations (PDEs) is
that of physics-informed neural networks (PINNs). Introduced in 2017 by Raissi,
Perdikaris, and Karniadakis, the method works by encoding the physics of the
problem into the loss function of a neural network. In particular, this means
that the loss function will usually be a sum of three terms:

1. One term enforcing the PDE
2. One term enforcing the boundary conditions
3. One term enforcing the initial conditions

The big idea is then that as the neural network is trained and this loss
function is minimized, the neural network will converge to the solution of the
boundary value problem.

### Boundary Value Problems

Before talking at all about PINNs, we first must discuss the problems we want
them to solve: boundary-value problems.
