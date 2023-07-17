use elara_log::prelude::*;
use elara_math::prelude::*;

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    let rand = Tensor::rand([3, 1]) * 2.0 - 1.0;
    println!("{:?}", rand);
}
