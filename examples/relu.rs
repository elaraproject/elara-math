use elara_math::prelude::*;

fn main() {
    let t1 = Tensor::arange(-10..10).reshape([2, 5]);
    println!("{:?}", t1);
    let t2 = t1.relu();
    println!("{:?}", t2);
}