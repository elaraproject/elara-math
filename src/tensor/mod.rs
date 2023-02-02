use std::fmt::Debug;

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
}
