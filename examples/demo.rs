use elara_math::Tensor;

fn main() {
    #[rustfmt::skip]
    let t = Tensor::new(&[
        1, 2, 3, 
        4, 5, 6], [2, 3]);

    println!("{}", t[&[1, 0]]);
}
