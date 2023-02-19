use elara_math::Tensor;

fn main() {
    let t1: Tensor<f64, 2> = Tensor::new(&[
    	1.0, 2.0,
    	3.0, 4.0], [2, 2]);
    let t2: Tensor<f64, 2> = Tensor::new(&[
    	5.0, 6.0,
    	7.0, 8.0], [2, 2]);
    println!("{:?}", t1);
    println!("{:?}", t2);
    // Expected: [[19, 22], [43, 50]]
    println!("{:?}", t1.matmul(&t2));
}