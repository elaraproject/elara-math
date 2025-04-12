/*! This crate is a Rust-native tensor and math library,
developed for [Project Elara](https://github.com/elaraproject). It aims to
be a basic implementation of common machine learing and scientific computing
tools in Rust. It tries to give a Rust experience similar to [SciPy](https://scipy.org/),
[NumPy](https://numpy.org/), and [PyTorch](https://pytorch.org/)/[TensorFlow](https://www.tensorflow.org/),
offering:

- Tensors: N-dimensional differentiable arrays (via `Tensor`)
- An implementation of reverse-mode automatic differentiation
- Numerical solvers for calculus (only numerical integration/quadrature is fully-supported at the moment, the ODE solver has been moved to [`elara-array`](https://github.com/elaraproject/elara-array))

To get started, just run `cargo install elara-math` to add to your project. We offer [several examples](https://github.com/elaraproject/elara-math/tree/main/examples) in our GitHub repository to reference and learn from.
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
