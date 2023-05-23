#[macro_use]
extern crate impl_ops;

mod num;
mod tensor;

pub use num::*;
pub use tensor::*;
// pub use autograd;


pub mod prelude;

// pub fn exp<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
//     let exp_vec = x.iter()
//         .map(|a| a.exp())
//         .collect();
//     Tensor {
//         shape: x.shape,
//         data: exp_vec
//     }
// }
//
// pub fn sigmoid<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
//     let sigmoid_vec = x.iter()
//         .map(|a| T::one() / (T::one() + (-a.clone()).exp()))
//         .collect();
//     Tensor {
//         shape: x.shape,
//         data: sigmoid_vec
//     }
// }
//
// pub fn sigmoid_d<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
//     let sigmoid_vec = x.iter()
//         .map(|a| a.clone() * (T::one() - a.clone()))
//         .collect();
//     Tensor {
//         shape: x.shape,
//         data: sigmoid_vec
//     }
// }
//
//
// pub fn sqrt<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
//     let sqrt_vec = x.iter()
//         .map(|a| a.clone().sqrt())
//         .collect();
//     Tensor {
//         shape: x.shape,
//         data: sqrt_vec
//     }
// }
//
// pub fn sin<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
// 	let sin_vec = x.iter()
// 		.map(|a| a.clone().sin())
// 		.collect();
// 	Tensor {
// 		shape: x.shape,
// 		data: sin_vec
// 	}
// }
//
// pub fn cos<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
// 	let cos_vec = x.iter()
// 		.map(|a| a.clone().cos())
// 		.collect();
// 	Tensor {
// 		shape: x.shape,
// 		data: cos_vec
// 	}
// }
//
// pub fn tanh<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Float
// {
// 	let tanh_vec = x.iter()
// 		.map(|a| a.clone().tanh())
// 		.collect();
// 	Tensor {
// 		shape: x.shape,
// 		data: tanh_vec
// 	}
// }
//
// pub fn maximum<T, const N: usize>(x: &Tensor<T, N>, y: &Tensor<T, N>) -> Tensor<T, N>
// where T: Clone + PartialOrd
// {
// 	assert_eq!(x.shape, y.shape);
// 	let max_vec = x.iter()
// 		.zip(&y.data)
// 		.map(|(a, b)| num::max(a.clone(), b.clone()))
// 		.collect();
// 	Tensor {
// 		shape: x.shape,
// 		data: max_vec
// 	}
// }
//
// pub fn relu<T, const N: usize>(x: &Tensor<T, N>) -> Tensor<T, N>
// where T: Zero + Clone + PartialOrd
// {
// 	let relu_vec = x.iter()
// 		.map(|a| num::max(T::zero(), a.clone()))
// 		.collect();
// 	Tensor {
// 		shape: x.shape,
// 		data: relu_vec
// 	}
// }
