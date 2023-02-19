use elara_log::prelude::*;
use elara_math::{Tensor, tensor};

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    let a: Tensor<i32, 1> = tensor!([1, 2, 3, 4, 5]);
    let b: Tensor<i32, 1> = tensor!([2, 3, 4, 5, 6]);
    let dot_prod = a.dot(&b);
    println!("Dot product: {:?}", dot_prod);
    println!("{:?}", a);
}
