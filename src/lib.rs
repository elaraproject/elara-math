mod num;
mod tensor;

pub use num::*;
pub use tensor::*;


pub mod prelude;

pub fn mse(predicted: &Tensor, target: &Tensor) -> Tensor
{
    let out = target - predicted;
    (&out * &out).mean()
}
