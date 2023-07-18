use elara_log::prelude::*;
use elara_math::prelude::*;
use std::time::Instant;

const EPOCHS: usize = 10000;
const LR: f64 = 1e-5;

fn forward_pass(data: &Tensor, weights: &Tensor, biases: &Tensor) -> Tensor {
    (&data.matmul(&weights) + &biases).relu()
}

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

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
    let weights = Tensor::rand([3, 1]);
    let biases = Tensor::rand([4, 1]);
    println!("Weights before training:\n{:?}", weights);
    let now = Instant::now();
    for epoch in 0..(EPOCHS + 1) {
        let output = forward_pass(&train_data, &weights, &biases);
        let loss = elara_math::mse(&output, &train_labels);
        println!("Epoch {}, loss {:?}", epoch, loss);
        loss.backward();
        weights.update(LR);
        weights.zero_grad();
        biases.update(LR);
        biases.zero_grad();
    }
    println!("{:?}", now.elapsed());
    let pred_data = tensor![[1.0, 0.0, 0.0]];
    let pred = forward_pass(&pred_data, &weights, &biases);
    println!("Weights after training:\n{:?}", weights);
    println!("Prediction [1, 0, 0] -> {:?}", pred);
}
