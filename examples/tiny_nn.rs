use elara_log::prelude::*;
use elara_math::prelude::*;

const EPOCHS: usize = 1000;
const LR: f64 = 1e-5;

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

    #[rustfmt::skip]
    let train_data: Tensor<2> = tensor![
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]];
    #[rustfmt::skip]
    let train_labels: Tensor<2> = tensor![
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ].reshape([4, 1]);
    let mut weights = Tensor::rand([3, 1]);
    println!("Weights before training: {:?}", weights);
    for epoch in 0..EPOCHS {
        let output = (&train_data.matmul(&weights)).sigmoid();
        let mut loss = elara_math::mse(&output, &train_labels);
        println!("Epoch {} loss: {:?}", epoch, loss);
        loss.backward();
        let adjustment = weights.grad() * LR;
        weights = &weights - &Tensor::new_from_f64(adjustment);
        weights.zero_grad();
    }
    let pred_data: Tensor<2> = tensor![[1.0, 0.0, 0.0]];
    let pred = &pred_data.matmul(&weights).sigmoid();
    println!("Weights after training: {:?}", weights);
    println!("Prediction [1, 0, 0] -> {:?}", pred.to_ndarray().data);
}
