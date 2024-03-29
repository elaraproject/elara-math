# TODO

- [x] `Tensor` basic class
- [x] Basic tensor creation methods:
    - [x] `ones()`
    - [x] `zeros()`
    - [x] `empty()`
    - [ ] `identity()`
    - [x] `random()`
- [x] Basic arithmetical operations
    - [x] Add
    - [x] Subtract
    - [x] Multiply
    - [x] Divide
    - [x] Negation
- [ ] Basic operations
    - [x] `reshape()`
    - [x] `arange()`
    - [ ] `linspace()`
    - [x] `flatten()`
    - [x] `max()`
    - [x] `min()`
    - [ ] `clip()`
    - [x] `sum()`
    - [x] `product()`
    - [x] `mean()`
    - [ ] `det()`
    - [ ] `append()` - should update shape correspondingly
    - [ ] `pop()` - should update shape correspondingly
    - [ ] `rescale()` (for normalizing values between two numbers, e.g. 0 to 1)
- [ ] Basic linear algebra
    - [x] Transpose
    - [x] Dot product
    - [x] Matrix product
    - [ ] Cross product
    - [ ] Inverse
    - [ ] Norm
- [x] Move all linear algebra operations to top-level of crate
- [x] Autograd
- [ ] More advanced scientific computing functionality
    - [ ] Linear equation solver
    - [ ] Curve fitting
    - [ ] Optimizer
    - [ ] Numerical integrator
    - [ ] ODE solver
- [ ] Refactoring
    - [x] Rewrite with generics
    - [ ] Implement `Debug` to print tensors
    - [ ] Improve performance with 100% reference-based implementation and reduced number of clones
- [ ] Benchmarks
