# Elara Math

Elara Math is a Rust-native math library, with (current or planned support for):

- Tensors (n-dimensional arrays)\* (supported)
- Linear algebra with tensors (supported)
- Numerical solvers for integration & differential equations (WIP)
- Automatic differentiation (supported)

\*: GPU tensors are not available yet, but GPU acceleration is planned to be added in the future

As an example, here is a working tiny neural network using `elara-math`.

```rs
use elara_math::prelude::*;

const EPOCHS: usize = 10000;
const LR: f64 = 0.01;

fn main() {
    let train_data = tensor![
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]];
    let train_labels = tensor![
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ].reshape([4, 1]);
    let mut weights = Tensor::rand([3, 1]);
    for epoch in 0..EPOCHS {
        let output = train_data.matmul(&weights).sigmoid();
        let loss = elara_math::mse(&output, &train_labels);
        println!("Epoch {}, loss: {:?}", epoch, loss);
        loss.backward();
        let adjustment = weights.grad() * LR;
        weights = weights - Tensor::new(adjustment);
        weights.zero_grad();
    }
    let pred_data = tensor![[1.0, 0.0, 0.0]];
    let pred = &pred_data.matmul(&weights).sigmoid();
    println!("Prediction [1, 0, 0] -> {:?}", pred.borrow().data);
}
```

## Developing

To develop `elara-math`, first clone the repository:

```
git clone https://github.com/elaraproject/elara-math
```

Then, copy over the pre-commit githook:

```
cp .githooks/pre-commit .git/hooks/pre-commit && chmod a+x .git/hooks/pre-commit
```

You should then be all set to start making changes!
