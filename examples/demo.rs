use elara_math::Tensor;
use elara_math::{n, a};

fn main() {
    let array = a(vec![
        a(vec![n(1.0), n(2.0), n(3.0)]), 
        a(vec![n(4.0), n(5.0), n(6.0)])
    ]);
    let tensor_a = Tensor::new(&array);
    println!("{:?}", tensor_a);
    println!("Shape: {:?}", tensor_a.shape());
}
