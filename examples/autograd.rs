
use elara_log::prelude::*;
use elara_math::prelude::*;

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    let mut x: Tensor<1> = tensor![3.0];
    let mut y = &x * &x;

    y.backward();
    println!("dy/dx: {:?}", x.grad().data);
}
