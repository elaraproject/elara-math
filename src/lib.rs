mod num;
mod tensor;

pub use num::*;
pub use tensor::*;
// pub use autograd;
// pub use linalg;

use num_traits::Float;

pub fn exp<T>(x: T) -> T 
where T: Float
{
    x.exp()
}