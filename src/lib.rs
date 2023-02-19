mod num;
mod tensor;

pub use num::*;
pub use tensor::*;
// pub use autograd;
// pub use linalg;

use num_traits::Float;

pub fn exp<T, const N: usize>(x: Tensor<T, N>) -> Tensor<T, N> 
where T: Float
{
    let exp_vec = x.iter()
        .map(|a| a.exp())
        .collect();
    Tensor {
        shape: x.shape,
        data: exp_vec
    }
}

pub fn sigmoid<T, const N: usize>(x: Tensor<T, N>) -> Tensor<T, N>
where T: Float
{
    let sigmoid_vec = x.iter()
        .map(|a| T::one() / (T::one() + (-a.clone()).exp()))
        .collect();
    Tensor {
        shape: x.shape,
        data: sigmoid_vec
    }
}

pub fn sigmoid_d<T, const N: usize>(x: Tensor<T, N>) -> Tensor<T, N>
where T: Float
{
    let sigmoid_vec = x.iter()
        .map(|a| a.clone() * (T::one() - a.clone()))
        .collect();
    Tensor {
        shape: x.shape,
        data: sigmoid_vec
    }
}