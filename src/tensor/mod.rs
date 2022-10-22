use std::convert::From;
use std::fmt::Debug;

pub use MultiArray::Array as a;
pub use MultiArray::Number as n;

// general GPU tensor type
#[derive(Debug)]
pub struct Tensor<'a, T> 
    where T: Clone + Copy + Debug + 'static
{
    buffer: &'a MultiArray<T>,
    shape: Vec<i32>,
    // todo: add gpu buffer
}

// Rust does not natively support multi-dimensional arrays
// so we create a type for it
#[derive(Clone, Debug)]
pub enum MultiArray<S>
    where S: Clone
{
    Number(S),
    Array(Vec<MultiArray<S>>)
}

impl<S: Clone> MultiArray<S> {
    fn len<T>(&self) -> Option<i32>
        where T: From<T> + Copy
    {
        if let MultiArray::Array(arr) = self {
            Some(arr.len() as i32)
        } else {
            None
        }
    }

    fn first_el<T>(&self) -> Option<&MultiArray<S>>
        where T: From<T> + Copy
    {
        if let MultiArray::Array(arr) = self {
            Some(&arr[0])
        } else {
            None
        }
    }
}

/* 
    Python equivalent:

    def dims(a):
    if not type(a) == list:
        return []
    return [len(a)] + dims(a[0])

    As clearly, dims() is absolutely
    impossible to understand esoteric code
    that tries to make Rust pretend to
    be like Python
*/
pub fn dims<T>(array: &MultiArray<T>) -> Vec<i32>
    where T: Clone + Copy + Debug
{
    if let MultiArray::Number(_i) = array {
        return [].to_vec();
    }
    let mut shape = vec![array.len::<T>().unwrap()];
    shape.extend(dims(array.first_el::<T>().unwrap()));
    shape
}

// TODO: Create tensor!() macro to turn vectors automatically
// into MultiArrays

impl<T: Clone + Copy + Debug> Tensor<'_, T> {
    /* 
        Ideal new function would use the tensor!()
        macro so that the buffer can just be a vector
        not a MultiArray.
        That should be what we implement next.
    */
    pub fn new(array: &MultiArray<T>) -> Tensor<T> {
        let array_shape = dims(array);
        Tensor {
         shape: array_shape,
         buffer: array
        }
    }

    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    pub fn buffer(&self) -> &MultiArray<T> {
        &self.buffer
    }

}
