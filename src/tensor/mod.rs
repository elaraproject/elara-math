mod array;
pub use array::{NdArray, utils::*};

use crate::array;

use std::{
    cell::{RefCell, Ref, RefMut},
    collections::HashSet,
    hash::{Hash, Hasher},
    fmt::Debug,
    ops::{Add, Mul, Div, Sub, Index, AddAssign, SubAssign, Deref, DerefMut}, rc::Rc,
    iter::Sum
};

use uuid::Uuid;

#[macro_export]
macro_rules! tensor {
    [$([$($x:expr),* $(,)*]),+ $(,)*] => {
        Tensor::new($crate::array!($([$($x,)*],)*))
    };
    [$($x:expr),*] => {
        Tensor::new($crate::array!($($x),*))
    };
}

#[macro_export]
macro_rules! scalar {
    ($x:expr) => {
        Tensor::from_f64($crate::array!($x))
    }
}

pub struct TensorData<const N: usize> {
    pub data: NdArray<f64, N>,
    pub grad: NdArray<f64, N>,
    pub uuid: Uuid,
    backward: Option<fn(&TensorData<N>)>,
    prev: Vec<Tensor<N>>,
    op: Option<String>
}

#[derive(Clone)]
pub struct Tensor<const N: usize>(Rc<RefCell<TensorData<N>>>);

impl<const N: usize> Hash for Tensor<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl<const N: usize> PartialEq for Tensor<N> {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl<const N: usize> Eq for Tensor<N> {}

impl<const N: usize> Deref for Tensor<N> {
    type Target = Rc<RefCell<TensorData<N>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for Tensor<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> TensorData<N> {
    fn new(data: NdArray<f64, N>) -> TensorData<N> {
        let shape = data.shape.clone();
        TensorData {
            data, 
            grad: NdArray::zeros(shape),
            uuid: Uuid::new_v4(),
            backward: None,
            prev: Vec::new(),
            op: None
        }
    }
}

impl<const N: usize> Tensor<N> {
    pub fn new(array: NdArray<f64, N>) -> Tensor<N> {
        Tensor(Rc::new(RefCell::new(TensorData::new(array))))
    }

    pub fn shape(&self) -> [usize; N] {
        self.borrow().data.shape
    }

    pub fn rand(shape: [usize; N]) -> Tensor<N> {
        let arr = NdArray::random(shape);
        Tensor::new(arr)
    }

    pub fn from_f64(val: f64) -> Tensor<N> {
        Tensor::new(array!(val))
    }

    pub fn grad(&self) -> NdArray<f64, N> {
        self.borrow().grad.clone()
    }

    pub fn reshape(&mut self, shape: [usize; N]) -> Tensor<N> {
        Tensor::new(self.borrow().data.clone().reshape(shape))
    }
    
    pub fn index(&self, idx: &[usize; N]) -> f64 {
        self.borrow().data[idx]
    }
    
    pub fn len(&self) -> usize {
        self.borrow().data.len()
    }
    
    pub fn sum(&self) -> Tensor<N> {
        let sum = self.borrow().data.sum();
        let out = Tensor::from_f64(sum);
        out.borrow_mut().prev = vec![self.clone()];
        out.borrow_mut().op = Some(String::from("sum"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            // let shape = value.prev[0].borrow().data.shape.clone();
            // value.prev[0].borrow_mut().grad += value.grad.clone() * NdArray::ones(shape);
            value.prev[0].borrow_mut().grad += value.grad.clone();
        });
        out
    }
    
    pub fn mean(&self) -> Tensor<N> {
        let len = Tensor::from_f64(self.len() as f64);
        self.sum() / len
    }
    
    pub fn exp(&self) -> Tensor<N> {
        let exp_array = self.borrow().data.mapv(|val| val.exp());
        let out = Tensor::new(exp_array);
        out.borrow_mut().prev = vec![self.clone()];
        out.borrow_mut().op = Some(String::from("exp"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            let prev = value.prev[0].borrow().data.clone();
            value.prev[0].borrow_mut().grad += prev.mapv(|val| val.exp());
        });
        out
    }
    
    pub fn pow(&self, power: f64) -> Tensor<N> {
        let pow_array = self.borrow().data.mapv(|val| val.powf(power));
        let out = Tensor::new(pow_array);
        out.borrow_mut().prev = vec![self.clone(), Tensor::from_f64(power)];
        out.borrow_mut().op = Some(String::from("^"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            let base = value.prev[0].borrow().data.clone();
            let p = value.prev[1].borrow().data.clone();
            let base_vec = base.mapv(|val| val.powf(p.first().unwrap() - 1.0));
            value.prev[0].borrow_mut().grad += p * base_vec * value.grad.clone();
        });
        out
    }
    
    pub fn sigmoid(&self) -> Tensor<N> {
        let sigmoid_array = self.borrow().data.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        let out = Tensor::new(sigmoid_array);
        out.borrow_mut().prev = vec![self.clone()];
        out.borrow_mut().op = Some(String::from("exp"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            let prev = value.prev[0].borrow().data.clone();
            value.prev[0].borrow_mut().grad += prev.mapv(|val| val.exp() / (1.0 + val.exp()).powf(2.0));
        });
        out
    }
    
    pub fn index_mut(&mut self, idx: &[usize; N]) -> f64 {
        self.borrow_mut().data[idx]
    }

    // pub fn data(&self) -> impl Deref<Target = NdArray<f64, N>> + '_ {
    //     Ref::map((*self.0).borrow(), |mi| &mi.data)
    // }
    
    // pub fn data_mut(&self) -> impl Deref<Target = NdArray<f64, N>> + '_ {
    //     RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.data)
    // }

    // pub fn grad(&self) -> impl Deref<Target = NdArray<f64, N>> + '_ {
    //     Ref::map((*self.0).borrow(), |mi| &mi.grad)
    // }

    pub fn grad_mut(&self) -> impl DerefMut<Target = NdArray<f64, N>> + '_ {
        RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.grad)
    }

    pub fn backward(&self) {
        let mut topo: Vec<Tensor<N>> = vec![];
        let mut visited: HashSet<Tensor<N>> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo.reverse();

        self.grad_mut().fill(1.0);
        for v in topo {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        }
    }

    fn _build_topo(&self, topo: &mut Vec<Tensor<N>>, visited: &mut HashSet<Tensor<N>>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }
}

// TODO: better printing of tensors
impl<const N: usize> Debug for Tensor<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, shape={:?})", self.borrow().data.clone(), self.shape())
    }
}

impl Tensor<2> {
    pub fn matmul(&self, rhs: &Tensor<2>) -> Tensor<2> {
        let a_shape = self.shape();
        let b_shape = rhs.shape();
        assert_eq!(a_shape[1], b_shape[0]);
    	let res: NdArray<f64, 2> = self.borrow().data.matmul(&rhs.borrow().data);
        let out = Tensor::new(res);
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("matmul"));
        out.borrow_mut().backward = Some(|value: &TensorData<2>| {
            let a_data = value.prev[0].borrow().data.clone();
            let b_data = value.prev[1].borrow().data.clone();
            value.prev[0].borrow_mut().grad += b_data * value.grad.clone();
            value.prev[1].borrow_mut().grad += a_data * value.grad.clone();
        });
        out
    }
}

// Elementwise addition by reference
impl<const N: usize> Add<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;
    fn add(self, rhs: &Tensor<N>) -> Self::Output {
        let out = Tensor::new(self.borrow().data.clone() + rhs.borrow().data.clone());
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("+"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            value.prev[0].borrow_mut().grad += value.grad.clone();
            value.prev[1].borrow_mut().grad += value.grad.clone();
        });
        out
    }
}

// Elementwise addition without reference
impl<const N: usize> Add<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;
    fn add(self, rhs: Tensor<N>) -> Self::Output {
        &self + &rhs
    }
}

// Elementwise subtraction by reference
impl<const N: usize> Sub<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;
    fn sub(self, rhs: &Tensor<N>) -> Self::Output {
        let out = Tensor::new(self.borrow().data.clone() + rhs.borrow().data.clone());
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("-"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            value.prev[0].borrow_mut().grad -= value.grad.clone();
            value.prev[1].borrow_mut().grad -= value.grad.clone();
        });
        out
    }
}


// Elementwise multiplication without reference
impl<const N: usize> Mul<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;

    fn mul(self, rhs: &Tensor<N>) -> Self::Output {
        let out = Tensor::new(self.borrow().data.clone() * rhs.borrow().data.clone());
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("×"));
        out.borrow_mut().backward = Some(|value: &TensorData<N>| {
            let a_data = value.prev[0].borrow().data.clone();
            let b_data = value.prev[1].borrow().data.clone();
            value.prev[0].borrow_mut().grad += b_data * value.grad.clone();
            value.prev[1].borrow_mut().grad += a_data * value.grad.clone();
        });
        out
    }
}

// Elementwise multiplication without reference
impl<const N: usize> Mul<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;

    fn mul(self, rhs: Tensor<N>) -> Self::Output {
        &self * &rhs
    }
}

// Elementwise division without reference
impl<const N: usize> Div<&Tensor<N>> for &Tensor<N> {
    type Output = Tensor<N>;
    
    fn div(self, rhs: &Tensor<N>) -> Self::Output {
        self * &rhs.pow(-1.0)
    }
}

impl<const N: usize> Div<Tensor<N>> for Tensor<N> {
    type Output = Tensor<N>;
    
    fn div(self, rhs: Tensor<N>) -> Self::Output {
        &self / &rhs
    }
}
