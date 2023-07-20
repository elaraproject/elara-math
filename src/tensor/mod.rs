use elara_log::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// use crate::randf;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
    rc::Rc,
};

use uuid::Uuid;

/// A macro for counting the number of args
/// passed to it
#[macro_export]
macro_rules! count {
    [$($x:expr),*] => {
        vec![$($x),*].len()
    }
}

/// Macro for quickly creating tensors
#[macro_export]
macro_rules! tensor {
    [$([$($x:expr),* $(,)*]),+ $(,)*] => {
        Tensor::new(ndarray::array!($([$($x,)*],)*))
    };
    [$($x:expr),*] => {
        Tensor::new(ndarray::array!($($x),*).into_shape(($crate::count![$($x),*], 1)).unwrap())
    };
}


/// Macro for quickly creating scalar tensors
#[macro_export]
macro_rules! scalar {
    ($x:expr) => {
        Tensor::from_f64($x)
    };
}

/// Backing data for `Tensor`
pub struct TensorData {
    pub data: Array2<f64>,
    pub grad: Array2<f64>,
    pub uuid: Uuid,
    backward: Option<fn(&TensorData)>,
    prev: Vec<Tensor>,
    op: Option<String>,
}

/// A PyTorch-like differentiable tensor type
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
    fn new(data: Array2<f64>) -> TensorData {
        let shape = data.raw_dim();
        TensorData {
            data,
            grad: Array2::zeros(shape),
            uuid: Uuid::new_v4(),
            backward: None,
            prev: Vec::new(),
            op: None,
        }
    }
}

impl Tensor {
    /// Create a new tensor from an `Array2`
    pub fn new(array: Array2<f64>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData::new(array))))
    }

    /// Find the shape of a tensor
    pub fn shape(&self) -> (usize, usize) {
        self.borrow().data.dim()
    }

    /// Create a tensor filled with random values
    pub fn rand(shape: [usize; 2]) -> Tensor {
        let arr: Array2<f64> = Array2::random((shape[0], shape[1]), Uniform::new(0., 1.));
        Tensor::new(arr)
    }

    /// Create a tensor from a `f64`
    pub fn from_f64(val: f64) -> Tensor {
        Tensor::new(array![[val]])
    }

    /// Create a tensor of shape filled with ones
    pub fn ones(shape: [usize; 2]) -> Tensor {
        let arr: Array2<f64> = Array2::ones((shape[0], shape[1]));
        Tensor::new(arr)
    }

    /// Create a tensor of shape filled with zeros
    pub fn zeros(shape: [usize; 2]) -> Tensor {
        let arr: Array2<f64> = Array2::zeros((shape[0], shape[1]));
        Tensor::new(arr)
    }

    /// Update tensor value given its derivative
    /// and a learning rate; useful for machine learning
    /// applications
    pub fn update(&self, lr: f64) {
        let mut data = self.inner_mut();
        let grad = data.grad.clone();
        data.data.scaled_add(-lr, &grad);
    }

    /// Create a tensor from a range
    pub fn arange<I: Iterator<Item = i32>>(range: I, shape: [usize; 2]) -> Tensor {
        let arr = Array::from_iter(range).mapv(|el| el as f64).into_shape((shape[0], shape[1])).unwrap();
        Tensor::new(arr)
    }

    /// Create a tensor containing a linearly-spaced
    /// interval
    pub fn linspace(start: f64, end: f64, num: usize) -> Tensor {
        let arr = Array::linspace(start, end, num);
        let arr_reshaped = arr.into_shape((num, 1)).unwrap();
        Tensor::new(arr_reshaped)
    }

    /// Change the shape of a tensor and return a new tensor
    pub fn reshape(&mut self, shape: [usize; 2]) -> Tensor {
        Tensor::new(self.data().clone().into_shape(shape).unwrap())
    }

    /// Get a value from a tensor by index
    // pub fn index(&self, idx: &[usize; 2]) -> f64 {
    //     self.borrow().data[idx]
    // }

    /// Get the number of elements in a tensor
    pub fn len(&self) -> usize {
        self.data().len()
    }

    /// Find the sum of a tensor
    pub fn sum(&self) -> Tensor {
        let sum = self.data().sum();
        let out = Tensor::from_f64(sum);
        out.inner_mut().prev = vec![self.clone()];
        out.inner_mut().op = Some(String::from("sum"));
        out.inner_mut().backward = Some(|value: &TensorData| {
            // let shape = value.prev[0].data().raw_dim();
            // let ones: Array2<f64> = Array2::ones(shape).mapv(|el| el as f64);
            // let grad_value = value.grad[[0, 0]];
            // value.prev[0].grad_mut().add_assign(ones * grad_value);

            // value.prev[0].grad_mut().scaled_add(1.0, &value.prev[0].inner().grad)
            // let shape = value.prev[0].data().raw_dim();
            // let ones: Array2<f64> = Array2::ones(shape).mapv(|el: i32| el as f64);
            // let grad_value = value.grad[[0, 0]];
            value.prev[0].grad_mut().scaled_add(1.0, &value.grad);
        });
        out
    }

    /// Find the mean of a tensor
    pub fn mean(&self) -> Tensor {
        (1.0 / self.data().len() as f64) * self.sum()
    }

    /// Exponential function for tensors
    pub fn exp(&self) -> Tensor {
        let exp_array = self.borrow().data.mapv(|val| val.exp());
        let out = Tensor::new(exp_array);
        out.inner_mut().prev = vec![self.clone()];
        out.inner_mut().op = Some(String::from("exp"));
        out.inner_mut().backward = Some(|value: &TensorData| {
            let prev = value.prev[0].borrow().data.clone();
            value.prev[0].grad_mut().scaled_add(1.0, &prev.mapv(|val| val.exp()));
        });
        out
    }

    /// ReLU function for tensors
    pub fn relu(&self) -> Tensor {
        let relu_array = self.data().mapv(|val| val.max(0.0));
        let out = Tensor::new(relu_array);
        out.inner_mut().prev = vec![self.clone()];
        out.inner_mut().op = Some(String::from("ReLU"));
        out.inner_mut().backward = Some(|value: &TensorData| {
            let dv = value.prev[0].data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            value.prev[0].grad_mut().scaled_add(1.0, &dv);
        });
        out
    }

    // WARNING: power function breaks easily and is hacked together with bits
    // and pieces from a soul that is haunted with weeks of midnight code
    // NEEDS TO BE REWRITTEN!!!

    /// Power function for tensors (not recommended as it breaks easily)
    pub fn pow(&self, power: f64) -> Tensor {
        warn!("pow() is not yet workable at the moment");
        let pow_array = self.data().mapv(|val| val.powf(power));
        let out = Tensor::new(pow_array);
        out.inner_mut().prev = vec![self.clone(), Tensor::from_f64(power)];
        out.inner_mut().op = Some(String::from("^"));
        out.inner_mut().backward = Some(|value: &TensorData| {
            let base_vec = value.prev[0].data().mapv(|val| val.powf(value.prev[1].data()[[0, 0]] - 1.0));
            value.prev[0].grad_mut().scaled_add(1.0, &(value.prev[1].data().deref() * base_vec * value.grad.clone()));
        });
        out
    }

    /// Sigmoid function for tensors (not recommended as well)
    pub fn sigmoid(&self) -> Tensor {
        let sigmoid_array = self.borrow().data.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        let out = Tensor::new(sigmoid_array);
        out.inner_mut().prev = vec![self.clone()];
        out.inner_mut().op = Some(String::from("exp"));
        out.inner_mut().backward = Some(|value: &TensorData| {
            let prev = value.prev[0].borrow().data.clone();
            let exp_array = prev.mapv(|val| val.exp() / (1.0 + val.exp()).powf(2.0));
            value.prev[0].inner_mut().grad.scaled_add(1.0, &exp_array);
        });
        out
    }

    /// Tensor matrix multiplication
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let a_shape = self.shape();
        let b_shape = rhs.shape();
        if a_shape.1 != b_shape.0 {
            error!("You are attempting to matrix-multiply two matrices of size {} x {} and {} x {}. These shapes are not compatible.", a_shape.0, a_shape.1, b_shape.0, b_shape.1);
        }
        let res: Array2<f64> = self.data().dot(rhs.data().deref());
        let out = Tensor::new(res);
        out.inner_mut().prev = vec![self.clone(), rhs.clone()];
        out.inner_mut().op = Some(String::from("matmul"));
        out.inner_mut().backward = Some(|value: &TensorData| {
            let da = value.grad.dot(&value.prev[1].data().t());
            let db = value.prev[0].data().t().dot(&value.grad);
            value.prev[0].grad_mut().scaled_add(1.0, &da);
            value.prev[1].grad_mut().scaled_add(1.0, &db);
        });
        out
    }

    /// Get an element of a tensor as mutable
    // pub fn index_mut(&mut self, idx: &[usize; 2]) -> f64 {
    //     self.inner_mut().data[idx]
    // }

    /// Get the underlying `TensorData` of a tensor
    pub fn inner(&self) -> Ref<TensorData> {
        (*self.0).borrow()
    }

    /// Get the underlying `TensorData` of a tensor
    /// as mutable
    pub fn inner_mut(&self) -> RefMut<TensorData> {
        (*self.0).borrow_mut()
    }

    /// Get the underlying data NdArray of a tensor
    pub fn data(&self) -> impl Deref<Target = Array2<f64>> + '_ {
        Ref::map((*self.0).borrow(), |mi| &mi.data)
    }

    /// Get the underlying data NdArray of a tensor
    /// as mutable
    pub fn data_mut(&self) -> impl DerefMut<Target = Array2<f64>> + '_ {
        RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.data)
    }

    /// Find the gradient of a tensor
    /// Remember to call `backward()` first!
    pub fn grad(&self) -> impl Deref<Target = Array2<f64>> + '_ {
        Ref::map((*self.0).borrow(), |mi| &mi.grad)
    }

    /// Get the gradient of a tensor as mutable
    /// Remember to call `backward()` first!
    pub fn grad_mut(&self) -> impl DerefMut<Target = Array2<f64>> + '_ {
        RefMut::map((*self.0).borrow_mut(), |mi| &mut mi.grad)
    }

    /// Zero the gradient of a tensor
    pub fn zero_grad(&self) {
        self.grad_mut().fill(0.0);
    }

    /// Perform backpropagation on a tensor
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

    // Thanks to: https://stackoverflow.com/questions/76727378/how-to-implement-iter-for-a-type-that-wraps-an-ndarray
    pub fn iter(&self) -> impl Iterator<Item = Tensor> + '_ {
        let data = self.data();
        (0..data.shape()[0]).map(move |i| {
            let el = data.index_axis(Axis(0), i);
            let reshaped_and_cloned_el = el
                .into_shape((el.shape()[0], 1))
                .unwrap()
                .mapv(|el| el.clone());
            Tensor::new(reshaped_and_cloned_el)
        })
    }
}

impl Iterator for Tensor {
    type Item = Tensor;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.iter().next().unwrap()) 
    }
}

// TODO: better printing of tensors
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}",
            self.data().deref(),
        )
    }
}

macro_rules! impl_binary_op {
    [$trait:ident, $op_name:ident, $op:tt] => {
        impl $trait for Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: Tensor) -> Self::Output {
                &self $op &rhs
            }
        } 

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: f64) -> Self::Output {
                self $op &Tensor::from_f64(rhs)
            }
        }

        impl $trait<f64> for Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: f64) -> Self::Output {
                &self $op rhs
            }
        }

         impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $op_name(self, rhs: &Tensor) -> Self::Output {
                &Tensor::from_f64(self) $op rhs
            }
        }

        impl $trait<Tensor> for f64 {
            type Output = Tensor;

            fn $op_name(self, rhs: Tensor) -> Self::Output {
                self $op &rhs
            }
        }
    };

    [$trait:ident, $op_name:ident, $op:tt, $update_grad:expr] => {
        impl $trait for &Tensor {
            type Output = Tensor;

            fn $op_name(self, rhs: &Tensor) -> Self::Output {
                let out = Tensor::new(self.data().deref() $op rhs.data().deref());
                out.inner_mut().prev = vec![self.clone(), rhs.clone()];
                out.inner_mut().op = Some(stringify!($op_name).to_string());
                out.inner_mut().backward = Some(|value: &TensorData| {
                    let (dv1, dv2) = $update_grad(&value.grad, value.prev[0].data().deref(), value.prev[1].data().deref());

                    let dv1 = match value.prev[0].grad().dim() {
                        (1, 1) => arr2(&[[dv1.sum()]]),
                        (1, n) => dv1.sum_axis(Axis(0)).into_shape((1, n)).unwrap(),
                        (n, 1) => dv1.sum_axis(Axis(1)).into_shape((n, 1)).unwrap(),
                        (_, _) => dv1,
                    };
                    let dv2 = match value.prev[1].grad().dim() {
                        (1, 1) => arr2(&[[dv2.sum()]]),
                        (1, n) => dv2.sum_axis(Axis(0)).into_shape((1, n)).unwrap(),
                        (n, 1) => dv2.sum_axis(Axis(1)).into_shape((n, 1)).unwrap(),
                        (_, _) => dv2,
                    };

                    value.prev[0].grad_mut().scaled_add(1.0, &dv1);
                    value.prev[1].grad_mut().scaled_add(1.0, &dv2);
                });
                out
            }
        }

        impl_binary_op![$trait, $op_name, $op];
    };
}

impl_binary_op![Add, add, +, |grad, _a, _b| { (grad * 1.0, grad * 1.0) }];
impl_binary_op![Sub, sub, -, |grad, _a, _b| { (grad * 1.0, grad * -1.0) }];
impl_binary_op![Mul, mul, *, |grad, a, b| { (grad * b, grad * a) }];
impl_binary_op![Div, div, /, |grad, a, b| { (grad * 1.0 / b, grad * -1.0 * a / (b * b)) }];

// // Elementwise multiplication without reference
// impl Mul<&Tensor> for &Tensor {
//     type Output = Tensor;
// 
//     fn mul(self, rhs: &Tensor) -> Self::Output {
//         let out = Tensor::new(self.data().deref() * rhs.data().deref());
//         out.inner_mut().prev = vec![self.clone(), rhs.clone()];
//         out.inner_mut().op = Some(String::from("×"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             // |grad, a, b| { (grad * b, grad * a) }
//             let (dv1, dv2) = {(
//                 &value.grad * value.prev[1].data().deref(), &value.grad * value.prev[0].data().deref()
//             )};
//             value.prev[0].grad_mut().scaled_add(1.0, &dv1);
//             value.prev[1].grad_mut().scaled_add(1.0, &dv2);
//             // let mut a_data = value.prev[0].inner_mut();
//             // let mut b_data = value.prev[1].inner_mut();
//             // a_data.grad.scaled_add(1.0, &(&b_data.data * &value.grad));
//             // b_data.grad.scaled_add(1.0, &(&a_data.data * &value.grad));
//         });
//         out
//     }
// }
// // Elementwise addition by reference
// impl Add<&Tensor> for &Tensor {
//     type Output = Tensor;
//     fn add(self, rhs: &Tensor) -> Self::Output {
//         let out = Tensor::new(self.data().deref() + rhs.data().deref());
//         out.inner_mut().prev = vec![self.clone(), rhs.clone()];
//         out.inner_mut().op = Some(String::from("+"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             value.prev[0].grad_mut().scaled_add(1.0, &value.grad);
//             value.prev[1].grad_mut().scaled_add(1.0, &value.grad);
// 
//         });
//         out
//     }
// }
// 
// // Elementwise addition without reference
// impl Add<Tensor> for Tensor {
//     type Output = Tensor;
//     fn add(self, rhs: Tensor) -> Self::Output {
//         &self + &rhs
//     }
// }
// 
// // Scalar addition by reference
// impl Add<f64> for &Tensor {
//     type Output = Tensor;
//     fn add(self, rhs: f64) -> Self::Output {
//         let out = Tensor::new(self.data().deref() + rhs);
//         out.inner_mut().prev = vec![self.clone(), Tensor::from_f64(rhs)];
//         out.inner_mut().op = Some(String::from("+"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             value.prev[0].grad_mut().scaled_add(1.0, &value.grad);
//             value.prev[1].grad_mut().scaled_add(1.0, &value.grad);
//         });
//         out
//         
//     }
// }
// 
// // Scalar addition without reference
// impl Add<f64> for Tensor {
//     type Output = Tensor;
//     fn add(self, rhs: f64) -> Self::Output {
//         &self + rhs
//     }
// }
// 
// // Elementwise subtraction by reference
// impl Sub<&Tensor> for &Tensor {
//     type Output = Tensor;
//     fn sub(self, rhs: &Tensor) -> Self::Output {
//         let out = Tensor::new(self.data().deref() - rhs.data().deref());
//         out.inner_mut().prev = vec![self.clone(), rhs.clone()];
//         out.inner_mut().op = Some(String::from("-"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             value.prev[0].grad_mut().scaled_add(-1.0, &value.grad);
//             value.prev[1].grad_mut().scaled_add(-1.0, &value.grad);
//         });
//         out
//     }
// }
// 
// // Elementwise subtraction without reference
// impl Sub<Tensor> for Tensor {
//     type Output = Tensor;
//     fn sub(self, rhs: Tensor) -> Self::Output {
//         &self - &rhs
//     }
// }
// 
// // Scalar subtraction by reference
// impl Sub<f64> for &Tensor {
//     type Output = Tensor;
//     fn sub(self, rhs: f64) -> Self::Output {
//         let out = Tensor::new(self.data().deref() - rhs);
//         out.inner_mut().prev = vec![self.clone(), Tensor::from_f64(rhs)];
//         out.inner_mut().op = Some(String::from("-"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             let dv = arr2(&[[value.grad.sum()]]);
//             value.prev[0].grad_mut().scaled_add(-1.0, &dv);
//             value.prev[1].grad_mut().scaled_add(-1.0, &dv);
//         });
//         out
//         
//     }
// }
// 
// // Scalar subtraction without reference
// impl Sub<f64> for Tensor {
//     type Output = Tensor;
//     fn sub(self, rhs: f64) -> Self::Output {
//         &self - rhs
//     }
// }
// 
// 
// // Elementwise multiplication with reference
// impl Mul<Tensor> for Tensor {
//     type Output = Tensor;
// 
//     fn mul(self, rhs: Tensor) -> Self::Output {
//         &self * &rhs
//     }
// }
// 
// // Elementwise division without reference
// impl Div<&Tensor> for &Tensor {
//     type Output = Tensor;
// 
//     fn div(self, rhs: &Tensor) -> Self::Output {
//         let out = Tensor::new(self.data().deref() / rhs.data().deref());
//         out.inner_mut().prev = vec![self.clone(), rhs.clone()];
//         out.inner_mut().op = Some(String::from("/"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             let a_data = value.prev[0].data().clone();
//             let b_data = value.prev[1].data().clone();
//             let a2_data = a_data.clone();
//             let b2_data = b_data.clone();
//             value.prev[0].grad_mut().scaled_add(1.0, &(-&value.grad / (a_data * a2_data)));
//             value.prev[1].grad_mut().scaled_add(1.0, &(-&value.grad / (b_data * b2_data)));
//         });
//         out
//     }
// }
// 
// // Scalar multiplication by reference
// impl Mul<f64> for &Tensor {
//     type Output = Tensor;
// 
//     fn mul(self, rhs: f64) -> Self::Output {
//         let out = Tensor::new(self.data().deref() * rhs);
//         out.inner_mut().prev = vec![self.clone(), Tensor::from_f64(rhs)];
//         out.inner_mut().op = Some(String::from("×"));
//         out.inner_mut().backward = Some(|value: &TensorData| {
//             let (mut dv1, mut dv2) = {(
//                 &value.grad * value.prev[1].data().deref(), &value.grad * value.prev[0].data().deref()
//             )};
//             dv1 = arr2(&[[dv1.sum()]]);
//             dv2 = arr2(&[[dv2.sum()]]);
//             value.prev[0].grad_mut().scaled_add(1.0, &dv1);
//             value.prev[1].grad_mut().scaled_add(1.0, &dv2);
//         });
//         out
//     }
// }
// 
// // Scalar multiplication without reference
// impl Mul<f64> for Tensor {
//     type Output = Tensor;
// 
//     fn mul(self, rhs: f64) -> Self::Output {
//         &self * rhs
//     }
// }
// 
// impl Div<Tensor> for Tensor {
//     type Output = Tensor;
// 
//     fn div(self, rhs: Tensor) -> Self::Output {
//         &self / &rhs
//     }
// }
