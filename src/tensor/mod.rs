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

pub struct TensorData {
    pub data: NdArray<f64, 2>,
    pub grad: NdArray<f64, 2>,
    pub uuid: Uuid,
    backward: Option<fn(&TensorData)>,
    prev: Vec<Tensor>,
    op: Option<String>
}

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Eq for Tensor {}

impl Deref for Tensor {
    type Target = Rc<RefCell<TensorData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TensorData {
    fn new(data: NdArray<f64, 2>) -> TensorData {
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

impl Tensor {
    pub fn new(array: NdArray<f64, 2>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData::new(array))))
    }

    pub fn shape(&self) -> [usize; 2] {
        self.borrow().data.shape
    }

    pub fn rand(shape: [usize; 2]) -> Tensor {
        let arr = NdArray::random(shape);
        Tensor::new(arr)
    }

    pub fn from_f64(val: f64) -> Tensor {
        Tensor::new(array![[val]])
    }

    pub fn grad(&self) -> NdArray<f64, 2> {
        self.borrow().grad.clone()
    }

    pub fn reshape(&mut self, shape: [usize; 2]) -> Tensor {
        Tensor::new(self.borrow().data.clone().reshape(shape))
    }
    
    pub fn index(&self, idx: &[usize; 2]) -> f64 {
        self.borrow().data[idx]
    }
    
    pub fn len(&self) -> usize {
        self.borrow().data.len()
    }
    
    pub fn sum(&self) -> Tensor {
        let sum = self.borrow().data.sum();
        let out = Tensor::from_f64(sum);
        out.borrow_mut().prev = vec![self.clone()];
        out.borrow_mut().op = Some(String::from("sum"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            let shape = value.prev[0].borrow().data.shape.clone();
            value.prev[0].borrow_mut().grad += NdArray::ones(shape) * value.grad.first().unwrap().clone();
            // value.prev[0].borrow_mut().grad += value.grad.clone();
        });
        out
    }
    
    pub fn mean(&self) -> Tensor {
        let len = Tensor::from_f64(self.len() as f64);
        self.sum() / len
    }
    
    pub fn exp(&self) -> Tensor {
        let exp_array = self.borrow().data.mapv(|val| val.exp());
        let out = Tensor::new(exp_array);
        out.borrow_mut().prev = vec![self.clone()];
        out.borrow_mut().op = Some(String::from("exp"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            let prev = value.prev[0].borrow().data.clone();
            value.prev[0].borrow_mut().grad += prev.mapv(|val| val.exp());
        });
        out
    }

    // WARNING: power function breaks easily and is hacked together with bits
    // and pieces from a soul that is haunted with weeks of midnight code
    // NEEDS TO BE REWRITTEN!!!
    pub fn pow(&self, power: f64) -> Tensor {
        let pow_array = self.borrow().data.mapv(|val| val.powf(power));
        let out = Tensor::new(pow_array);
        out.borrow_mut().prev = vec![self.clone(), Tensor::from_f64(power)];
        out.borrow_mut().op = Some(String::from("^"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            let base = value.prev[0].borrow().data.clone();
            let p = value.prev[1].borrow().data.clone();
            let base_vec = base.mapv(|val| val.powf(p.first().unwrap() - 1.0));
            value.prev[0].borrow_mut().grad += p * base_vec * value.grad.clone();
        });
        out
    }
    
    pub fn sigmoid(&self) -> Tensor {
        let sigmoid_array = self.borrow().data.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        let out = Tensor::new(sigmoid_array);
        out.borrow_mut().prev = vec![self.clone()];
        out.borrow_mut().op = Some(String::from("exp"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            let prev = value.prev[0].borrow().data.clone();
            value.prev[0].borrow_mut().grad += prev.mapv(|val| val.exp() / (1.0 + val.exp()).powf(2.0));
        });
        out
    }


    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let a_shape = self.shape();
        let b_shape = rhs.shape();
        assert_eq!(a_shape[1], b_shape[0]);
    	let res: NdArray<f64, 2> = self.borrow().data.matmul(&rhs.borrow().data);
        let out = Tensor::new(res);
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("matmul"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            let lhs = value.prev[0].borrow().data.clone();
            let rhs = value.prev[1].borrow().data.clone();
            let da = value.grad.clone().matmul(&rhs.transpose());
            let db = lhs.transpose().matmul(&value.grad.clone());
            value.prev[0].borrow_mut().grad += da;
            value.prev[1].borrow_mut().grad += db;
        });
        out
    }
    
    pub fn index_mut(&mut self, idx: &[usize; 2]) -> f64 {
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

    pub fn grad_mut(&self) -> impl DerefMut<Target = NdArray<f64, 2>> + '_ {
        RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.grad)
    }

    pub fn zero_grad(&self) {
        self.grad_mut().fill(0.0);
    }

    pub fn backward(&self) {
        let mut topo: Vec<Tensor> = vec![];
        let mut visited: HashSet<Tensor> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo.reverse();

        self.grad_mut().fill(1.0);
        for v in topo {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        }
    }

    fn _build_topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<Tensor>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }
}

// TODO: better printing of tensors
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, shape={:?})", self.borrow().data.data.clone(), self.shape())
    }
}


// Elementwise addition by reference
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        
        let out = Tensor::new(self.borrow().data.clone() + rhs.borrow().data.clone());
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("+"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            value.prev[0].borrow_mut().grad += value.grad.clone();
            value.prev[1].borrow_mut().grad += value.grad.clone();
        });
        out
    }
}

// Elementwise addition without reference
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        &self + &rhs
    }
}

// Elementwise subtraction by reference
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let out = Tensor::new(self.borrow().data.clone() - rhs.borrow().data.clone());
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("-"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            value.prev[0].borrow_mut().grad -= value.grad.clone();
            value.prev[1].borrow_mut().grad -= value.grad.clone();
        });
        out
    }
}

// Elementwise subtraction without reerence
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        &self - &rhs
    }
}


// Elementwise multiplication without reference
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let out = Tensor::new(self.borrow().data.clone() * rhs.borrow().data.clone());
        out.borrow_mut().prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut().op = Some(String::from("×"));
        out.borrow_mut().backward = Some(|value: &TensorData| {
            let a_data = value.prev[0].borrow().data.clone();
            let b_data = value.prev[1].borrow().data.clone();
            value.prev[0].borrow_mut().grad += b_data * value.grad.clone();
            value.prev[1].borrow_mut().grad += a_data * value.grad.clone();
        });
        out
    }
}

// Elementwise multiplication without reference
impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        &self * &rhs
    }
}

// Elementwise division without reference
impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    
    fn div(self, rhs: &Tensor) -> Self::Output {
        self * &rhs.pow(-1.0)
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    
    fn div(self, rhs: Tensor) -> Self::Output {
        &self / &rhs
    }
}
