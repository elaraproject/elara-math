use elara_math::Tensor;
use elara_math::E;

fn main() {
    let train_data = Tensor::new(&[0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], [4, 3]);
    let train_labels = Tensor::new(&[0, 1, 1, 0], [4, 1]);

    // Use Tensor::random once I get that working
    let weights: Tensor<i32, 2> = Tensor::zeros([3, 1]);
}
