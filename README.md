# Monolith

[![CI](https://github.com/elijahkin/monolith/actions/workflows/ci.yml/badge.svg)](https://github.com/elijahkin/monolith/actions/workflows/ci.yml)

> _One Repo to rule them all, One Repo to find them_\
> _One Repo to bring them all and in the GitHub bind them._

A C++23 monorepo of computational math and systems projects, built with
[Bazel](https://bazel.build).

## Projects

| Project                         | Description                                                                        | Dependencies                                        |
| ------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------- |
| [`collatz`](collatz/)           | Collatz conjecture graph visualization                                             | GraphViz                                            |
| [`instructions`](instructions/) | XLA-like computation graph optimizer with algebraic rewrites                       | —                                                   |
| [`mendel`](mendel/)             | Genetic algorithm framework with tournament selection and elitism                  | —                                                   |
| [`numerics`](numerics/)         | Numerical library: BLAS, FFT, ODE solvers, polynomials, rootfinding, number theory | —                                                   |
| [`simdelbrot`](simdelbrot/)     | AVX-512 + OpenMP accelerated Mandelbrot set renderer (PNG)                         | [libpng](http://www.libpng.org/pub/png/libpng.html) |
| [`spla`](spla/)                 | Sparse linear algebra tensor library for PDEs                                      | —                                                   |
| [`tourney`](tourney/)           | Chess engine: game state representation, minimax search, perft                     | —                                                   |

All projects require a C++23 compiler and
[googletest](https://github.com/google/googletest) for tests.

## Building

```sh
bazel build //...
```

## Testing

```sh
bazel test //...
```
