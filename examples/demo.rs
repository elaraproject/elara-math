use elara_log::prelude::*;
use elara_math::Tensor;

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    #[rustfmt::skip]
    let t1 = Tensor::new(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0], [2, 3]);
    let t2 = t1 * 3.0;

    // println!("{:?}", t1);
    println!("{:?}", t2);
    // println!("{}", t1[&[1, 0]]);
}
