# Elara Math

Elara Math is a Rust-native math library, with (current or planned support for):

- Tensors (n-dimensional arrays)\* (supported)
- Linear algebra with tensors (supported)
- Numerical solvers for integration & differential equations (WIP)
- Automatic differentiation (WIP)

\*: GPU tensors are not available yet, but GPU acceleration is planned to be added in the future

As an example, here is a working tiny neural network using `elara-math`.

```rs
use elara_math::{Tensor, sigmoid, sigmoid_d};

fn main() {
    let train_data = Tensor::new(&[
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0], [4, 3]);
    let train_labels = Tensor::new(&[
    	0.0,
    	1.0,
    	1.0,
    	0.0], [1, 4]).t();
    let mut weights = Tensor::random([3, 1]) * 2.0 - 1.0;
    for _ in 0..10000 {
        let output = sigmoid(&train_data.matmul(&weights));
        let error = &train_labels - &output;
        let m = error * sigmoid_d(&output);
        let adjustment = train_data.t().matmul(&m);
        weights = weights + adjustment;
    }
    let pred_data: Tensor<f64, 2> = Tensor::new(&[1.0, 0.0, 0.0], [1, 3]);
    let pred = sigmoid(&pred_data.matmul(&weights));
    println!("Prediction [1, 0, 0] -> {:?}", pred.data[0]);
}
```

## Developing

To develop `elara-math`, first clone the repository:

```
git clone https://github.com/elaraproject/elara-gfx
```

Then, copy over the pre-commit githook:

```
cp .githooks/pre-commit .git/hooks/pre-commit && chmod a+x .git/hooks/pre-commit
```

You should then be all set to start making changes!
