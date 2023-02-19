use elara_log::prelude::*;
use elara_math::{tensor, Tensor};

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    #[rustfmt::skip]
    let t1 = Tensor::new(&[
        1, 2, 3,
        4, 5, 6], [2, 3]);
    // let t2= Tensor::ones([2, 3]);
    let t2 = Tensor::ones([2, 3]);
    let t3 = &t1 + &t2;
    let t4 = &t1 * 3;

    let t5 = Tensor::arange(0..4).reshape([2, 2]);

    let t6: Tensor<i32, 1> = tensor!([1, 2, 3, 4]);
    let t7 = tensor!([[3, 6], [9, 12]]);

    let t8 = &t5 + &t1;

    println!("{:?}", t1);
    println!("{:?}", t2);
    println!("{}", t1[&[1, 0]]);
    println!("{:?}", t3);
    println!("Dot product test: {:?}", t4.dot(&t1));
    println!("{:?}", t4);
    println!("Sum test: {:?}", t6.sum());
    println!("{:?}", t7);
    println!("{:?}", t8);
}
