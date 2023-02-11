use elara_log::prelude::*;
use std::convert::{From, Into};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub},
};

// general tensor (multi-dimensional array) type
#[derive(Debug, Clone)]
pub struct Tensor<const N: usize> {
    shape: [usize; N],
    data: Vec<f64>,
}

impl<const N: usize> Tensor<N> {
    pub fn new(array: &[f64], shape: [usize; N]) -> Self {
        Tensor {
            shape,
            data: array.to_vec(),
        }
    }

    pub fn from(array: Vec<f64>, shape: [usize; N]) -> Self {
        Tensor { shape, data: array }
    }

    pub fn zeros(shape: [usize; N]) -> Self {
        let mut i = 1;
        for j in 0..shape.len() {
            i *= shape[j]
        }
        Tensor {
            shape,
            data: vec![0.into(); i],
        }
    }

    pub fn ones(shape: [usize; N]) -> Self {
        let mut i = 1;
        for j in 0..shape.len() {
            i *= shape[j]
        }
        Tensor {
            shape,
            data: vec![1.into(); i],
        }
    }

    pub fn empty(shape: [usize; N]) -> Self {
        Tensor {
            shape,
            data: Vec::new(),
        }
    }

    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
    fn get_index(&self, idx: &[usize; N]) -> Result<usize, String> {
        let mut i = 0;
        for j in 0..self.shape.len() {
            if idx[j] >= self.shape[j] {
                let err = format!(
                    "[elara-math] Index {} is out of bounds for dimension {} with size {}",
                    idx[j], j, self.shape[j]
                );
                error!("{}", err);
                return Err(err);
            }
            i = i * self.shape[j] + idx[j];
        }
        Ok(i)
    }

    // Add and create new tensor
    fn add(self, other: Tensor<N>) -> Result<Tensor<N>, String> {
        if self.shape() != other.shape() {
            let err = format!(
                "[elara-math] Cannot two tensors of different shape {:?}, {:?}",
                self.shape(),
                other.shape()
            );
            error!("{}", err);
            return Err(err);
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor {
            shape: self.shape(),
            data: data,
        })
    }

    fn subtract(self, other: Tensor<N>) -> Result<Tensor<N>, String> {
        if self.shape() != other.shape() {
            let err = format!(
                "[elara-math] Cannot two tensors of different shape {:?}, {:?}",
                self.shape(),
                other.shape()
            );
            error!("{}", err);
            return Err(err);
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Tensor {
            shape: self.shape(),
            data: data,
        })
    }

    fn scalar_mul(self, scalar: f64) -> Tensor<N> {
        let data = self.data.iter().map(|el| el * scalar).collect();

        Tensor {
            shape: self.shape(),
            data: data,
        }
    }

    pub fn dot(self, other: Tensor<N>) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

// Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
impl<const N: usize> Index<&[usize; N]> for Tensor<N> {
    type Output = f64;

    fn index(&self, idx: &[usize; N]) -> &f64 {
        let i = self.get_index(&idx).unwrap();
        &self.data[i]
    }
}

impl<const N: usize> IndexMut<&[usize; N]> for Tensor<N> {
    fn index_mut(&mut self, idx: &[usize; N]) -> &mut f64 {
        let i = self.get_index(idx).unwrap();
        &mut self.data[i]
    }
}

impl<const N: usize> Add<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;

    fn add(self, rhs: Tensor<N>) -> Self::Output {
        self.add(rhs).unwrap()
    }
}

impl<const N: usize> Sub<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;

    fn sub(self, rhs: Tensor<N>) -> Self::Output {
        self.subtract(rhs).unwrap()
    }
}

impl<const N: usize> Mul<f64> for Tensor<N> {
    type Output = Tensor<N>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scalar_mul(rhs)
    }
}
