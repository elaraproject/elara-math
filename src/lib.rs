/*! This crate is a Rust-native tensor and math library,
developed for [Project Elara](https://github.com/elaraproject). It aims to
be a building block of a comprehensive machine learing and scientific computing
library in Rust. It tries to give a Rust experience similar to [SciPy](https://scipy.org/),
[NumPy](https://numpy.org/), and [PyTorch](https://pytorch.org/)/[TensorFlow](https://www.tensorflow.org/),
offering:

- Tensors: N-dimensional differentiable arrays (via `Tensor`)
- An implementation of reverse-mode autodifferentiation
- Numerical solvers for calculus (only numerical integral evaluation is fully-supported at the moment, the ODE solver has been moved to [`elara-array`](https://github.com/elaraproject/elara-array)):
*/

mod integrate;
mod nn;
mod tensor;

pub use nn::*;
pub use tensor::*;

/// `elara-math` prelude
pub mod prelude;

/// Mean squared error function
pub fn mse(predicted: &Tensor, target: &Tensor) -> Tensor {
    let out = target - predicted;
    (&out * &out).mean()
}
