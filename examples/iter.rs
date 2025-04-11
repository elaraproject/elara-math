use elara_math::prelude::*;

fn main() {
    let a = tensor![[1., 2.], [3., 4.]];
    for el in a.iter() {
        println!("{:?}", el);
    }
}
