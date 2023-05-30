use elara_log::prelude::*;
use std::iter::{Product, Sum};
use std::ops::{AddAssign, SubAssign};
use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};
use crate::num::randf;

pub mod utils;
use utils::{One, Zero};

/// Macro for quickly creating 1D or 2D arrays
#[macro_export]
macro_rules! array {
    [$([$($x:expr),* $(,)*]),+ $(,)*] => {{
        $crate::NdArray::from_vec2(vec![$([$($x,)*],)*])
    }};
    [$($x:expr),* $(,)*] => {{
        $crate::NdArray::from_vec1(vec![$($x,)*])
    }};
}

/// A general NdArray (multi-dimensional differentiable
/// array type)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct NdArray<T: Clone, const N: usize> {
    pub shape: [usize; N],
    pub data: Vec<T>,
}

impl<T: Clone, const N: usize> NdArray<T, N> {
    /// Creates a new NdArray from an array of
    /// values with a given shape
    pub fn new(array: &[T], shape: [usize; N]) -> Self {
        NdArray {
            shape,
            data: array.to_vec(),
        }
    }

    /// Creates a new NdArray with a `Vec` of
    /// values with a given shape
    pub fn from(array: Vec<T>, shape: [usize; N]) -> Self {
        NdArray { shape, data: array }
    }

    /// Creates a new NdArray filled with
    /// only zeroes
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Clone + Zero,
    {
        NdArray {
            shape,
            data: vec![T::zero(); shape.iter().product()],
        }
    }

    /// Creates a new NdArray filled with
    /// only ones
    pub fn ones(shape: [usize; N]) -> Self
    where
        T: Clone + One,
    {
        NdArray {
            shape,
            data: vec![T::one(); shape.iter().product()],
        }
    }

    /// Creates a new NdArray of a shape without
    /// specifying values
    pub fn empty(shape: [usize; N]) -> Self {
        NdArray {
            shape,
            data: Vec::new(),
        }
    }

    pub fn from_vec1(array: Vec<T>) -> Self {
        let shape = [array.len(); N];
        NdArray { shape, data: array }
    }

    pub fn from_vec2(array: Vec<[T; N]>) -> NdArray<T, 2>
    where
        T: Debug,
    {
        let mut shape: [usize; 2] = [0; 2];
        shape[0] = array.len();
        shape[1] = array[0].len();
        let flattened_arr: Vec<T> = array.into_iter().flatten().collect();
        NdArray {
            shape,
            data: flattened_arr,
        }
    }

    /// Finds the number of elements present
    /// in a NdArray
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Allows for iterating through elements
    /// of a NdArray
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
    
    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }
    
    pub fn mapv<B, F>(&self, f: F) -> NdArray<B, N>
    where T: Clone,
          F: FnMut(T) -> B,
          B: Clone
    {
        let data = self.data.clone();
        NdArray {
            data: data.into_iter().map(f).collect(),
            shape: self.shape
        }
    }

    pub fn fill(&mut self, val: T) {
        self.data = vec![val; self.shape.iter().product()]
    }

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
    fn get_index(&self, idx: &[usize; N]) -> usize {
        let mut i = 0;
        for j in 0..self.shape.len() {
            if idx[j] >= self.shape[j] {
                error!("[elara-math] Index {} is out of bounds for dimension {} with size {}", idx[j], j, self.shape[j])
            }
            i = i * self.shape[j] + idx[j];
        }
        i
    }

    /// Change the shape of a NdArray
    pub fn reshape(self, shape: [usize; N]) -> NdArray<T, N> {
        if self.len() != shape.iter().product() {
            error!("[elara-math] Cannot reshape into provided shape {:?}", shape);
        }
        NdArray::from(self.data, shape)
    }

    /// Convert a higher-dimensional NdArray into
    /// a 1D NdArray
    pub fn flatten(self) -> NdArray<T, 1> {
        NdArray {
            data: self.data,
            shape: [self.shape.iter().product(); 1],
        }
    }
    

    /// Find the dot product of a NdArray with
    /// another NdArray
    pub fn dot(&self, other: &NdArray<T, N>) -> T
    where
        T: Clone + Zero + Mul<Output = T>
    {
        if self.len() != other.len() {
            error!("[elara-math] Dot product cannot be found between NdArrays of shape {} and {}, consider using matmul()",
        self.len(), other.len())
        }
        let mut product = T::zero();
        for i in 0..self.len() {
            product = product + self.data[i].clone() + other.data[i].clone()
        }
        product
    }

    /// Create a NdArray from a range of values
    pub fn arange<I: Iterator<Item = T>>(range: I) -> NdArray<T, N> {
        let vec: Vec<T> = range.collect();
        let len = vec.len();
        NdArray::from(vec, [len; N])
    }

    /// Transposes a NdArray in-place
    pub fn transpose(mut self) -> NdArray<T, N> {
        self.shape.reverse();
        self
    }

    /// Transposes a NdArray and returns a new NdArray
    pub fn t(&self) -> NdArray<T, N> {
        let mut shape = self.shape;
        shape.reverse();
        let data = self.data.clone();
        
        NdArray {
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
        self.data.iter().cloned().sum()
    }

    pub fn product(&self) -> T
    where
        T: Clone + Product,
    {
        self.data.iter().cloned().product()
    }

    pub fn mean(&self) -> T
    where
        T: Clone + Sum + Div<usize, Output = T>,
    {
        self.sum() / self.len()
    }
}

impl<const N: usize> NdArray<f64, N> {
    /// Creates a new NdArray filled with
    /// random values
    pub fn random(shape: [usize; N]) -> Self {
        // There's GOT to be a more efficient way to do this
        let empty_vec = vec![0.0; shape.iter().product()];
        let data = empty_vec.iter().map(|_| randf()).collect();
        NdArray {
            shape,
            data
        }
    }
}

impl NdArray<f64, 2> {
    /// Finds the matrix product of 2 matrices
    pub fn matmul(&self, b: &NdArray<f64, 2>) -> NdArray<f64, 2> {
    	assert_eq!(self.shape[1], b.shape[0]);
    	let mut res: NdArray<f64, 2> = NdArray::zeros([self.shape[0], b.shape[1]]);
    	for row in 0..self.shape[0] {
    		for col in 0..b.shape[1] {
    			for el in 0..b.shape[0] {
    				res[&[row, col]] += self[&[row, el]] * b[&[el, col]]
    			}
    		}
    	}
        res
    }
}

// Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
impl<T: Clone, const N: usize> Index<&[usize; N]> for NdArray<T, N> {
    type Output = T;

    fn index(&self, idx: &[usize; N]) -> &T {
        let i = self.get_index(idx);
        &self.data[i]
    }
}

impl<T: Clone, const N: usize> IndexMut<&[usize; N]> for NdArray<T, N> {
    fn index_mut(&mut self, idx: &[usize; N]) -> &mut T {
        let i = self.get_index(idx);
        &mut self.data[i]
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<&NdArray<T, N>> for &NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn add(self, rhs: &NdArray<T, N>) -> Self::Output {
        if self.shape != rhs.shape {
            let err = format!(
                "[elara-math] Cannot add two NdArrays of differing shapes {:?}, {:?}",
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

        NdArray::from(sum_vec, self.shape)
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<NdArray<T, N>> for NdArray<T, N>
{
    type Output = NdArray<T, N>;
    
    fn add(self, rhs: NdArray<T, N>) -> Self::Output {
        &self + &rhs
    }
}

// Scalar addition
impl<T: Clone + Add<Output = T>, const N: usize> Add<T> for &NdArray<T, N> {
    type Output = NdArray<T, N>;
    
    fn add(self, val: T) -> Self::Output {
        let sum_vec = self
            .data
            .iter()
            .map(|a| a.clone() + val.clone())
            .collect();

        NdArray::from(sum_vec, self.shape)
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> Add<T> for NdArray<T, N> {
    type Output = NdArray<T, N>;
    
    fn add(self, val: T) -> Self::Output {
        &self + val
    }
}

// Scalar addassign

impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<T> for NdArray<T, N> {

    fn add_assign(&mut self, rhs: T) {
        let sum_vec = self
            .data
            .iter()
            .map(|a| a.clone() + rhs.clone())
            .collect();
        self.data = sum_vec;
    }
}

// Elementwise addasign
impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<&NdArray<T, N>> for &mut NdArray<T, N> {
    fn add_assign(&mut self, rhs: &NdArray<T, N>) {
        let sum_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        self.data = sum_vec;
    }
}

impl<T: Clone + Add<Output = T>, const N: usize> AddAssign<NdArray<T, N>> for NdArray<T, N> {
    fn add_assign(&mut self, rhs: NdArray<T, N>) {
        let sum_vec: Vec<T> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        self.data = sum_vec;
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<&NdArray<T, N>> for &NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn sub(self, rhs: &NdArray<T, N>) -> Self::Output {
        if self.shape != rhs.shape {
            let err = format!(
                "[elara-math] Cannot subtract two NdArrays of differing shapes {:?}, {:?}",
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

        NdArray::from(difference_vec, self.shape)
    }
}

// Elementwise subassign
impl<T: Clone + Sub<Output = T>, const N: usize> SubAssign<&NdArray<T, N>> for &mut NdArray<T, N> {
    fn sub_assign(&mut self, rhs: &NdArray<T, N>) {
        let sub_vec = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        self.data = sub_vec;
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> SubAssign<NdArray<T, N>> for NdArray<T, N> {
    fn sub_assign(&mut self, rhs: NdArray<T, N>) {
        let sub_vec: Vec<T> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        self.data = sub_vec;
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<NdArray<T, N>> for NdArray<T, N>
{
    type Output = NdArray<T, N>;
    
    fn sub(self, rhs: NdArray<T, N>) -> Self::Output {
        &self - &rhs
    }
}

// Scalar subtraction
impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for &NdArray<T, N> {
    type Output = NdArray<T, N>;
    
    fn sub(self, val: T) -> Self::Output {
        let sub_vec = self
            .data
            .iter()
            .map(|a| a.clone() - val.clone())
            .collect();

        NdArray::from(sub_vec, self.shape)
    }
}

impl<T: Clone + Sub<Output = T>, const N: usize> Sub<T> for NdArray<T, N> {
    type Output = NdArray<T, N>;
    
    fn sub(self, val: T) -> Self::Output {
        &self - val
    }
}

// Scalar multiplication
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for &NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn mul(self, val: T) -> Self::Output {
        let mul_vec = self.data.iter().map(|a| val.clone() * a.clone()).collect();

        NdArray::from(mul_vec, self.shape)
    }
}

impl<T: Clone + Mul<Output = T>, const N: usize> Mul<T> for NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn mul(self, val: T) -> Self::Output {
        (&self).mul(val)
    }
}

// Elementwise multiplication
impl<T: Clone + Mul<Output = T>, const N: usize> Mul<&NdArray<T, N>> for &NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn mul(self, rhs: &NdArray<T, N>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let mul_vec = self.data.iter().zip(&rhs.data).map(|(a, b)| a.clone() * b.clone()).collect();
        NdArray::from(mul_vec, self.shape)
    }
}

impl<T: Clone + Mul<Output = T>, const N: usize> Mul<NdArray<T, N>> for NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn mul(self, rhs: NdArray<T, N>) -> Self::Output {
        &self * &rhs
    }
}

// Scalar division
impl<T: Clone + Div<Output = T>, const N: usize> Div<T> for &NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn div(self, val: T) -> Self::Output {
        let quotient_vec = self.data.iter().map(|a| val.clone() / a.clone()).collect();

        NdArray::from(quotient_vec, self.shape)
    }
}

// Negation
impl<T: Clone + Neg<Output = T>, const N: usize> Neg for NdArray<T, N> {
    type Output = NdArray<T, N>;

    fn neg(mut self) -> Self::Output {
        for idx in 0..self.len() {
            self.data[idx] = -(self.data[idx].clone());
        }
        self
    }
}
