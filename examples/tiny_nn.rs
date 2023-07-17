use elara_log::prelude::*;
use elara_math::prelude::*;
use std::time::Instant;

const EPOCHS: usize = 9000;
const LR: f64 = 1e-5;

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
    let weights = Tensor::ones([3, 1]);
    println!("Weights before training:\n{:?}", weights);
    let now = Instant::now();
    for epoch in 0..(EPOCHS + 1) {
        let output = train_data.matmul(&weights).relu();
        let loss = elara_math::mse(&output, &train_labels);
        println!("Epoch {}, loss {:?}", epoch, loss);
        loss.backward();
        weights.update(LR);
        weights.zero_grad();
    }
    println!("{:?}", now.elapsed());
    let pred_data = tensor![[1.0, 0.0, 0.0]];
    let pred = &pred_data.matmul(&weights).relu();
    println!("Weights after training:\n{:?}", weights);
    println!("Prediction [1, 0, 0] -> {:?}", pred);
}
