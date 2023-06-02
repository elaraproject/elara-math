use elara_log::prelude::*;
use elara_math::{array, NdArray};

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    let a: NdArray<i32, 1> = array![1, 2, 3, 4, 5];
    let b: NdArray<i32, 1> = array![2, 3, 4, 5, 6];
    let dot_prod = a.dot(&b);
    println!("Dot product: {:?}", dot_prod);
    println!("{:?}", a);
}
