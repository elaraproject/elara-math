use elara_math::{Tensor, sigmoid, sigmoid_d, tensor};

const EPOCHS: usize = 10000;

fn main() {
    #[rustfmt::skip]
    let train_data = tensor![
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]];
    #[rustfmt::skip]
    let train_labels = tensor![
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ].reshape([4, 1]);
    let mut weights = Tensor::random([3, 1]) * 2.0 - 1.0;
    println!("Weights before training: {:?}", weights);
    for _ in 0..EPOCHS {
        let output = sigmoid(&train_data.matmul(&weights));
        let error = &train_labels - &output;
        let m = error * sigmoid_d(&output);
        let adjustment = train_data.t().matmul(&m);
        weights = weights + adjustment;
    }
    let pred_data: Tensor<f64, 2> = Tensor::new(&[1.0, 0.0, 0.0], [1, 3]);
    let pred = sigmoid(&pred_data.matmul(&weights));
    println!("Weights after training: {:?}", weights);
    println!("Prediction [1, 0, 0] -> {:?}", pred.data[0]);
}
