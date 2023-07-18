/*! This crate is a Rust-native tensor and math library that eventually aims to
be a building block of a comprehensive machine learing and scientific computing
library in Rust. It tries to be the analogue of [SciPy](https://scipy.org/),
[NumPy](https://numpy.org/), and [PyTorch](https://pytorch.org/),
offering:

- Differentiable tensors (via `Tensor`)
- An implementation of reverse-mode autodifferentiation
- Numerical solvers for a variety of numerical analysis problems (still WIP)
*/

mod num;
mod tensor;

pub use num::*;
pub use tensor::*;

/// `elara-math` prelude
pub mod prelude;

/// Mean squared error function
pub fn mse(predicted: &Tensor, target: &Tensor) -> Tensor {
    let out = target - predicted;
    (&out * &out).mean()
}
