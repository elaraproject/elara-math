use std::{fmt::Debug, ops::{IndexMut, Index, Add}, iter::Sum};

// general tensor (multi-dimensional array) type
#[derive(Debug)]
pub struct Tensor<T, const N: usize> {
    shape: [usize; N],
    data: Vec<T>,
}

impl<T, const N: usize> Tensor<T, N>
where
    T: Clone,
{
    pub fn new(array: &[T], shape: [usize; N]) -> Self {
        Tensor {
            shape,
            data: array.to_vec(),
        }
    }

    pub fn zeros(shape: [usize; N]) -> Self 
    where
        T: From<i32>
    {
        let mut i = 1;
        for j in 0..shape.len() {
            i *= shape[j]
        }
        Tensor {
            shape,
            data: vec![0.into(); i]
        }
    }

    pub fn ones(shape: [usize; N]) -> Self 
    where
        T: From<i32>
    {
        let mut i = 1;
        for j in 0..shape.len() {
            i *= shape[j]
        }
        Tensor {
            shape,
            data: vec![1.into(); i]
        }
    }

    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
    fn get_index(&self, idx: &[usize; N]) -> Result<usize, String> {
        let mut i = 0;
        for j in 0..self.shape.len() {
            if idx[j] >= self.shape[j] {
                return Err(format!("[elara-math] Index {} is out of bounds for dimension {} with size {}", idx[j], j, self.shape[j]))
            }
            i = i * self.shape[j] + idx[j];
        }
        Ok(i)
    }

    // fn add(&self, other: Tensor<T, N>) -> Result<Tensor<T, N>, String> 
    //     where &'static T: Add<&'static T> + 'static
    // {
    //     if self.shape() != other.shape() {
    //         return Err(format!("[elara-math] Cannot add tensors of different shapes {:?}, {:?}", self.shape(), other.shape()))
    //     }
    //     let first_data = self.data();
    //     let second_data = other.data();
    //     let mut sum_array = vec![0; self.len()];
    //     for (i, (a, b)) in first_data.iter().zip(&second_data).enumerate() {
    //         sum_array[i] = a + b;
    //     }
    //     let result = Tensor {
    //         shape: self.shape,
    //         data: sum_array
    //     };
    //     Ok(result)
    // }
}

    // Referenced https://codereview.stackexchange.com/questions/256345/n-dimensional-array-in-rust
impl<T, const N: usize> Index<&[usize; N]> for Tensor<T, N> 
where T: Clone
{
    type Output = T;

    fn index(&self, idx: &[usize; N]) -> &T {
        let i = self.get_index(&idx).unwrap();
        &self.data[i]
    }
}

impl<T, const N: usize> IndexMut<&[usize; N]> for Tensor<T, N> 
where T: Clone
{
    fn index_mut(&mut self, idx: &[usize; N]) -> &mut T {
        let i = self.get_index(idx).unwrap();
        &mut self.data[i]
    }
}