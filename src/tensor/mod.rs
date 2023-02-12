use elara_log::prelude::*;
use std::iter::Sum;
use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};

mod utils;
use utils::{One, Zero};

// general tensor (multi-dimensional array) type
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
    /// only zeroes
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

    /// Finds the number of elements present
    /// in a tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

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

    pub fn dot(self, other: &Tensor<T, N>) -> T
    where
        T: Clone + Mul<Output = T> + Sum,
    {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .sum()
    }

    pub fn arange<I: Iterator<Item = T>>(range: I) -> Tensor<T, N> {
        let vec: Vec<T> = range.collect();
        let len = vec.len();
        Tensor::from(vec, [len; N])
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

// Scalar multiplication
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for &Tensor<T, N> {
    type Output = Tensor<T, N>;

    fn mul(self, val: T) -> Self::Output {
        let mul_vec = self.data.iter().map(|a| val.clone() * a.clone()).collect();

        Tensor::from(mul_vec, self.shape.clone())
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
