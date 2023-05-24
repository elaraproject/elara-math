#[macro_use]
extern crate impl_ops;

mod num;
mod tensor;

pub use num::*;
pub use tensor::*;


pub mod prelude;

pub fn mse<const N: usize>(predicted: &Tensor<N>, target: &Tensor<N>) -> Tensor<N>
{
    Tensor::new(array!(((target - predicted).pow(2.0)).mean()))
}