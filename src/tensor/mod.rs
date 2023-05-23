mod array;
pub use array::{NdArray, utils::*};

mod autograd;
pub use autograd::Value;

#[macro_export]
macro_rules! tensor {
    ([$($x:expr),*] $(, $mth:ident = $val:expr)+) => {
        $crate::array!($($x),*).mapv(|elem: f64| {
            let mut value = val!(elem);
            $( value.$mth($val) )*;
            value
        })
    };
    [$($x:expr),*] => {
        $crate::array!($($x),*).mapv(|elem: f64| $crate::val!(elem))
    };
}

pub type Tensor<const N: usize> = NdArray<Value, N>;

impl<const N: usize> Tensor<N> {
    pub fn backward(&mut self) {
        self.mapv(|val: Value| val.backward());
    }
    
    pub fn grad(&mut self) -> NdArray<f64, N> {
        let data = self.data.clone();
        let gradients = data.into_iter().map(move |val| val.borrow().grad).collect();
        NdArray {
            data: gradients,
            shape: self.shape
        }
    }
}
