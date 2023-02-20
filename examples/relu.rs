use elara_math::{Tensor, relu};

fn main() {
    let t1: Tensor<i32, 2> = Tensor::arange(-10..10).reshape([2, 5]);
    println!("{:?}", t1);
    let t2 = relu(&t1);
    println!("{:?}", t2);
}