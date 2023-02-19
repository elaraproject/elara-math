use elara_log::prelude::*;
use std::iter::{Product, Sum};
use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};
use crate::num::randf;

mod utils;
use utils::{One, Zero};

/// Macro for quickly creating 1D, 2D, or 3D tensors
/// TODO: add support for 2D and 3D tensors
#[macro_export]
macro_rules! tensor {
    ([$([$($x:expr),* $(,)*]),+ $(,)*]) => {{
        Tensor::from_vec2(vec![$([$($x,)*],)*])
    }};
    ([$($x:expr),* $(,)*]) => {{
        Tensor::from_vec1(vec![$($x,)*])
    }};
}

/// A general tensor (multi-dimensional differentiable
/// array type)
#[derive(Debug, Clone)]
pub struct Tensor<T: Clone, const N: usize> {
    pub shape: [usize; N],
    pub data: Vec<T>,
}

impl<T: Clone, const N: usize> Tensor<T, N> {
    /// Creates a new tensor from an array of
    /// values with a given shape
    pub fn new(array: &[T], shape: [usize; N]) -> Self {
        Tensor {
            shape,
            data: array.to_vec(),
        }
    }

    /// Creates a new tensor with a `Vec` of
    /// values with a given shape
    pub fn from(array: Vec<T>, shape: [usize; N]) -> Self {
        Tensor { shape, data: array }
    }

    /// Creates a new tensor filled with
    /// only zeroes
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Clone + Zero,
    {
        Tensor {
            shape,
            data: vec![T::zero(); shape.iter().product()],
        }
    }

    /// Creates a new tensor filled with
    /// only ones
    pub fn ones(shape: [usize; N]) -> Self
    where
        T: Clone + One,
    {
        Tensor {
            shape,
            data: vec![T::one(); shape.iter().product()],
        }
    }

    /// Creates a new tensor of a shape without
    /// specifying values
    pub fn empty(shape: [usize; N]) -> Self {
        Tensor {
            shape,
            data: Vec::new(),
        }
    }

    pub fn from_vec1(array: Vec<T>) -> Self {
        let shape = [array.len(); N];
        Tensor { shape, data: array }
    }

    pub fn from_vec2(array: Vec<[T; N]>) -> Self
    where
        T: Debug,
    {
        let mut shape: [usize; N] = [2; N];
        shape[0] = array.len();
        shape[1] = array[0].len();
        let flattened_arr: Vec<T> = array.into_iter().flatten().collect();
        Tensor {
            shape,
            data: flattened_arr,
        }
    }

    /// Finds the number of elements present
    /// in a tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Allows for iterating through elements
    /// of a tensor
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
    fn get_index(&self, idx: &[usize; N]) -> usize {
        let mut i = 0;
        for j in 0..self.shape.len() {
            if idx[j] >= self.shape[j] {
                let err = format!(
                    "[elara-math] Index {} is out of bounds for dimension {} with size {}",
                    idx[j], j, self.shape[j]
                );
                error!("{}", err);
            }
            i = i * self.shape[j] + idx[j];
        }
        i
    }

    /// Change the shape of a tensor
    pub fn reshape(self, shape: [usize; N]) -> Tensor<T, N> {
        if self.len() != shape.iter().product() {
            let err = format!(
                "[elara-math] Cannot reshape into provided shape {:?}",
                shape
            );
            error!("{}", err);
        }
        Tensor::from(self.data, shape)
    }

    /// Convert a higher-dimensional tensor into
    /// a 1D tensor
    pub fn flatten(self) -> Tensor<T, 1> {
        Tensor {
            data: self.data,
            shape: [self.shape.iter().product(); 1],
        }
    }

    /// Find the dot product of a tensor with
    /// another tensor
    pub fn dot(&self, other: &Tensor<T, N>) -> T
    where
        T: Clone + Zero + Mul<Output = T>
    {
        assert_eq!(self.len(), other.len());
        let mut product = T::zero();
        for i in 0..self.len() {
            product = product + self.data[i].clone() + other.data[i].clone()
        }
        product
    }

    /// Matrix multiplication
    // pub fn matmul(&self, other: &Tensor<T, N>) -> Tensor<f64, N> {
    //     A_rows = self.len();
    // }

    /// Create a tensor from a range of values
    pub fn arange<I: Iterator<Item = T>>(range: I) -> Tensor<T, N> {
        let vec: Vec<T> = range.collect();
        let len = vec.len();
        Tensor::from(vec, [len; N])
    }

    pub fn transpose(mut self) -> Tensor<T, N> {
        self.shape.reverse();
        self
    }

    pub fn t(&self) -> Tensor<T, N> {
        let mut shape = self.shape.clone();
        shape.reverse();
        let data = self.data.clone();
        
        Tensor {
            shape,
            data
        }
    }

    pub fn max(&self) -> T
    where
        T: Ord,
    {
        self.data.iter().max().unwrap().clone()
    }

    pub fn min(&self) -> T
    where
        T: Ord,
    {
        self.data.iter().min().unwrap().clone()
    }

    pub fn sum(&self) -> T
    where
        T: Clone + Sum,
    {
        self.data.iter().map(|a| a.clone()).sum()
    }

    pub fn product(&self) -> T
    where
        T: Clone + Product,
    {
        self.data.iter().map(|a| a.clone()).product()
    }

    pub fn mean(&self) -> T
    where
        T: Clone + Sum + Div<usize, Output = T>,
    {
        self.sum() / self.len()
    }
}

impl<const N: usize> Tensor<f64, N> {

    /// Creates a new tensor filled with
    /// random values
    pub fn random(shape: [usize; N]) -> Self {
        // There's GOT to be a more efficient way to do this
        let empty_vec = vec![0.0; shape.iter().product()];
        let data = empty_vec.iter().map(|_| randf()).collect();
        Tensor {
            shape,
            data
        }
    }
}

// Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
impl<T: Clone, const N: usize> Index<&[usize; N]> for Tensor<T, N> {
    type Output = T;

    fn index(&self, idx: &[usize; N]) -> &T {
        let i = self.get_index(&idx);
        &self.data[i]
    }
}

impl<T: Clone, const N: usize> IndexMut<&[usize; N]> for Tensor<T, N> {
    fn index_mut(&mut self, idx: &[usize; N]) -> &mut T {
        let i = self.get_index(idx);
        &mut self.data[i]
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<&Tensor<T, N>> for &Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn add(self, rhs: &Tensor<T, N>) -> Self::Output {
        if self.shape != rhs.shape {
            let err = format!(
                "[elara-math] Cannot add two tensors of differing shapes {:?}, {:?}",
                self.shape, rhs.shape
            );
            error!("{}", err);
        }

        let sum_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Tensor::from(sum_vec, self.shape.clone())
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<Tensor<T, N>> for Tensor<T, N>
{
    type Output = Tensor<T, N>;
    
    fn add(self, rhs: Tensor<T, N>) -> Self::Output {
        &self + &rhs
    }
}

// Scalar addition
impl<T: Clone + Add<Output = T>, const N: usize> Add<T> for &Tensor<T, N> {
    type Output = Tensor<T, N>;
    
    fn add(self, val: T) -> Self::Output {
        let sum_vec = self
            .data
            .iter()
            .map(|a| a.clone() + val.clone())
            .collect();

        Tensor::from(sum_vec, self.shape.clone())
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<T> for Tensor<T, N> {
    type Output = Tensor<T, N>;
    
    fn add(self, val: T) -> Self::Output {
        &self + val
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<&Tensor<T, N>> for &Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn sub(self, rhs: &Tensor<T, N>) -> Self::Output {
        if self.shape != rhs.shape {
            let err = format!(
                "[elara-math] Cannot subtract two tensors of differing shapes {:?}, {:?}",
                self.shape, rhs.shape
            );
            error!("{}", err);
        }

        let difference_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Tensor::from(difference_vec, self.shape.clone())
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<Tensor<T, N>> for Tensor<T, N>
{
    type Output = Tensor<T, N>;
    
    fn sub(self, rhs: Tensor<T, N>) -> Self::Output {
        &self - &rhs
    }
}

// Scalar subtraction
impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for &Tensor<T, N> {
    type Output = Tensor<T, N>;
    
    fn sub(self, val: T) -> Self::Output {
        let sub_vec = self
            .data
            .iter()
            .map(|a| a.clone() - val.clone())
            .collect();

        Tensor::from(sub_vec, self.shape.clone())
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for Tensor<T, N> {
    type Output = Tensor<T, N>;
    
    fn sub(self, val: T) -> Self::Output {
        &self - val
    }
}

// Scalar multiplication
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for &Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn mul(self, val: T) -> Self::Output {
        let mul_vec = self.data.iter().map(|a| val.clone() * a.clone()).collect();

        Tensor::from(mul_vec, self.shape.clone())
    }
}

impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn mul(self, val: T) -> Self::Output {
        (&self).mul(val)
    }
}

// Scalar division
impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for &Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn div(self, val: T) -> Self::Output {
        let quotient_vec = self.data.iter().map(|a| val.clone() / a.clone()).collect();

        Tensor::from(quotient_vec, self.shape.clone())
    }
}

// Negation
impl<T: Clone + Neg<Output = T>, const N: usize> Neg for Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn neg(mut self) -> Self::Output {
        for idx in 0..self.len() {
            self.data[idx] = -(self.data[idx].clone());
        }
        self
    }
}
