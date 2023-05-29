#[macro_use]
extern crate impl_ops;

mod num;
mod tensor;

pub use num::*;
pub use tensor::*;


pub mod prelude;

pub fn mse<const N: usize>(predicted: &Tensor, target: &Tensor) -> Tensor
{
    (target - predicted).pow(2.0).mean()
}
