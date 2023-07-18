use elara_math::prelude::*;

fn main() {
    let x = Tensor::arange(-5..5, [10, 1]);
    let func = &x * -5.0;
    let y = &func + &(Tensor::rand([10, 1]) * 0.4);
    println!("{:?}", x);
    println!("{:?}", y);
}