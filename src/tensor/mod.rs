use std::{fmt::Debug, ops::{IndexMut, Index}};

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

    pub fn shape(&self) -> [usize; N] {
        self.shape
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
