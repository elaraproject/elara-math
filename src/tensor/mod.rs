mod array;
pub use array::{NdArray, utils::*};

mod autograd;
pub use autograd::Value;

use crate::num::randf;
use crate::val;

use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub, Index, AddAssign, SubAssign},
};

#[macro_export]
macro_rules! tensor {
    [$([$($x:expr),* $(,)*]),+ $(,)*] => {
        Tensor::new($crate::array!($([$($x,)*],)*).mapv(|elem: f64| $crate::val!(elem)))
    };
    [$($x:expr),*] => {
        Tensor::new($crate::array!($($x),*).mapv(|elem: f64| $crate::val!(elem)))
    };
}

pub struct Tensor<const N: usize>(NdArray<Value, N>);

impl<const N: usize> Tensor<N> {
    pub fn new(array: NdArray<Value, N>) -> Tensor<N> {
        Tensor(array)
    }

    pub fn new_from_f64(array: NdArray<f64, N>) -> Tensor<N> {
        Tensor(array.mapv(|val| val!(val)))
    }

    pub fn arange<I: Iterator<Item = i32>>(range: I) -> Tensor<N> {
        Tensor(NdArray::arange(range).mapv(|el| val!(el)))
    }

    pub fn zeros(shape: [usize; N]) -> Tensor<N> {
        let array = NdArray {
            shape,
            data: vec![val!(0.0); shape.iter().product()],
        };
        Tensor(array)
    }

    pub fn rand(shape: [usize; N]) -> Tensor<N> {
        let empty_vec = vec![0.0; shape.iter().product()];
        let data = empty_vec.iter().map(|_| crate::val!(randf())).collect();
        let array = NdArray {
            shape,
            data
        };
        Tensor(array)
    }

    pub fn backward(&mut self) {
        self.0.mapv(|val: Value| val.backward());
    }
    
    pub fn grad(&mut self) -> NdArray<f64, N> {
        let data = self.0.data.clone();
        let gradients = data.into_iter().map(move |val| val.borrow().grad).collect();
        NdArray {
            data: gradients,
            shape: self.0.shape
        }
    }

    pub fn set_data(&mut self, array: NdArray<Value, N>) {
        self.0 = array;
    }

    pub fn to_ndarray(&self) -> NdArray<f64, N> {
        self.0.mapv(|val| val.borrow().data)
    }

    pub fn zero_grad(&mut self) {
        self.0.mapv(|val| val.borrow_mut().zero_grad());
    }

    pub fn shape(&self) -> [usize; N] {
        self.0.shape.clone()
    }
    
    pub fn reshape(&mut self, shape: [usize; N]) -> Tensor<N> {
        let arr = NdArray {
            shape,
            data: self.0.data.clone()
        };
        Tensor(arr)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.0.data.iter()
    }

    pub fn len(&self) -> Value {
        val!(self.0.data.len() as f64)
    }

    pub fn sum(&self) -> Value {
        self.0.data.iter().cloned().sum()
    }
    
    pub fn mean(&self) -> Value {
        self.sum() / self.len()
    }

    pub fn exp(&self) -> Tensor<N> {
        Tensor(self.0.mapv(|val| val.exp()))
    }

    pub fn sigmoid(&self) -> Tensor<N> {
        Tensor(self.0.mapv(|val| 1.0 / (1.0 + (-val).exp())))
    }

    pub fn relu(&self) -> Tensor<N> {
        Tensor(self.0.mapv(|val| val.relu()))
    }

    pub fn pow(&self, base: f64) -> Tensor<N> {
        Tensor(self.0.mapv(|val| val.pow(base)))
    }
}

impl Tensor<2> {
    pub fn matmul(&self, b: &Tensor<2>) -> Tensor<2> {
    	assert_eq!(self.0.shape[1], b.0.shape[0]);
        let a_array = &self.0;
        let b_array = &b.0;
        let res = a_array.matmul(&b_array);
        let res: NdArray<Value, 2> = res.mapv(|val| val!(val));
        Tensor(res)
    }
}

impl<const N: usize> Debug for Tensor<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, shape={:?})", self.to_ndarray().data, self.shape())
    }
}

impl<const N: usize> Index<&[usize; N]> for Tensor<N> {
    type Output = Value;

    fn index(&self, index: &[usize; N]) -> &Self::Output {
        &self.0[index]
    }
}

// Elementwise addition by reference
impl<const N: usize> Add<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;
    fn add(self, rhs: &Tensor<N>) -> Self::Output {
        Tensor::new(&self.0 + &rhs.0)
    }
}

// Elementwise addition without reference
impl<const N: usize> Add<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;
    fn add(self, rhs: Tensor<N>) -> Self::Output {
        &self + &rhs
    }
}

// Elementwise addassign without reference
impl<const N: usize> AddAssign<Tensor<N>> for Tensor<N> {
    fn add_assign(&mut self, rhs: Tensor<N>) {
        let addassign_arr = &self.0 + &rhs.0;
        self.set_data(addassign_arr)
    }
}

// Elementwise subtraction by reference
impl<const N: usize> Sub<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;
    fn sub(self, rhs: &Tensor<N>) -> Self::Output {
        Tensor::new(&self.0 - &rhs.0)
    }
}

// Elementwise subtraction without reference
impl<const N: usize> Sub<&Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;
    fn sub(self, rhs: &Tensor<N>) -> Self::Output {
        &self - &rhs
    } 
}

// Elementwise subtract assign without reference
impl<const N: usize> SubAssign<Tensor<N>> for Tensor<N> {
    fn sub_assign(&mut self, rhs: Tensor<N>) {
        let subassign_arr = &self.0 - &rhs.0;
        self.set_data(subassign_arr)
    }
}

// Elementwise multiplication by reference
impl<const N: usize> Mul<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;
    fn mul(self, rhs: &Tensor<N>) -> Self::Output {
        Tensor::new(&self.0 * &rhs.0)
    }
}

// Elementwise multiplication without reference
impl<const N: usize> Mul<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;
    fn mul(self, rhs: Tensor<N>) -> Self::Output {
        &self * &rhs
    }
}