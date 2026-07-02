# SPLA

SPLA implements sparse linear algebra computations aimed toward numerical
solutions of partial differential equations. The fundamental object of
computation is the `spla::Tensor`, which generalizes the notions of scalars
(order-0), vectors (order-1), matrices (order-2), and higher order tensors. For
example usage, see `src/tensor_test.cc`.

## Work in Progress

The outstanding work is split broadly into the following three categories.

### Functionality

These tasks pertain to the implementation and correctness of mathematical
operations. The principal goal is to provide whatever operations our use cases
require.

- Diagnose and correct any failing tests.
- Add `eye` as a factory function.
- Ensure `operator[]` gracefully handles negative and out of bounds indices.
- Investigate whether the factory functions can easily be made free.
- Implement any remaining
  [operator overloads](https://cppreference.com/w/cpp/language/operators.html)
  and any
  [element-wise operations](https://cppreference.com/w/cpp/header/cmath.html)
  from the `<cmath>` header that seem relevant.
- Refactor `reduce` so that it returns a tensor and requires a list of reduction
  dimensions. This will allow for much more expressive reductions; for example,
  computing the row sums and column sums of matrices, rather than simply the sum
  of all entries.

### Code Quality

These tasks pertain to code elegance, readability, and documentation. The
principal goal is to ensure "self-documenting" code. That is, code which is so
clearly written that to comment on it would be redundant.

- Can we avoid redundant code in the constructors and `apply_binary_inplace`?
  Moreover, can we unify the constructors? Can we use `std::function`?
- Ensure thorough documentation of each class and function except those which
  are self-explanatory.
- Enforce good C++ practices: `#define` guards, long member functions outside
  class definitions.

### Performance Experiments

These tasks pertain to implementation details not relevant to the user. The
principal goal is to make the library as performant as possible.

- Think about whether it makes sense to allow writing `default_value_` to
  `data_` explicitly.
- Add parallelism with OpenMP. To do this we will need the compiler flag
  `-fopenmp`.
- Declaring `data_` as `std::unordered_map` instead of `std::map` bombards me
  with compile errors. Diagnose why this is and whether it would be difficult to
  fix. If not, do so and compare performance.
- Ensure `reduce` is designed such that `any` and `all` exit early.
