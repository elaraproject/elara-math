use elara_math::prelude::*;

fn main() {
    let x = scalar!(5.0);
    let y = x.pow(2.0);
    y.backward();
    println!("dy/dx: {:?}", x.grad().clone());
    x.zero_grad();
    let z = x.pow(-2.0);
    z.backward();
    println!("dz/dx: {:?}", x.grad().clone());
}
