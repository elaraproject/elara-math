# Elara Math

Elara Math is a Rust-native math library, developed as part of [Project Elara](https://github.com/elaraproject/)'s suite of open-source software libraries. It contains support for:

- Tensors: N-dimensional arrays with built-in support for [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation))\* (supported)
- Basic vectorized opertions and linear algebra with tensors (supported)
- Numerical solvers for integrals & differential equations (partially supported\*\*, work-in-progress)
- Basic machine learning tools for building feedforward fully-connected neural networks, with two APIs: one PyTorch-style, one TensorFlow-style (supported)

\*: GPU tensors are not available yet, but GPU acceleration is planned to be added in the future

\*\*: Numerical integration is supported out-of-the-box. However, the numerical differential equation solver has been (temporarily) moved to a separate library, [`elara-array`](https://github.com/elaraproject/elara-array), which is currently being developed in parallel.

> `elara-math` is **public domain software** like the rest of [Project Elara](https://github.com/elaraproject/), meaning it is essentially **unlicensed software**, so you can use it for basically any project you want, _however_ you want, with or without attribution.

**Shoutouts:** [Acknowledgements](https://github.com/elaraproject/elara-math/tree/main/ACKNOWLEDGEMENTS.md)

It is intended to both contain a set of ready-to-use solvers and vectorized math on NumPy-style N-dimensional arrays, as well as a flexible user-friendly API that can be used to make more specialized/advanced libraries for computational tasks.

As an example, here is a working tiny neural network using `elara-math` and its companion library [`elara-log`](https://github.com/elaraproject/elara-log) (`elara-log` is automatically installed when you install `elara-math`), ported from [this excellent Python demo](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1):

```rust
use elara_log::prelude::*;
use elara_math::prelude::*;

const EPOCHS: usize = 10000;
const LR: f64 = 1e-5;

fn forward_pass(data: &Tensor, weights: &Tensor, biases: &Tensor) -> Tensor {
    (&data.matmul(&weights) + biases).relu()
}

fn main() {
    // Initialize logging library
    Logger::new().init().unwrap();

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
    let weights = Tensor::rand([3, 1]);
    let biases = Tensor::rand([4, 1]);
    println!("Weights before training:\n{:?}", weights);
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
    let pred_data = tensor![[1.0, 0.0, 0.0]];
    let pred = forward_pass(&pred_data, &weights, &biases);
    println!("Weights after training:\n{:?}", weights);
    println!("Prediction [1, 0, 0] -> {:?}", pred);
}
```

For more examples, including basic usage of tensors, using automatic differentiation, and building more complex neural networks, please feel free to see the [examples folder](https://github.com/elaraproject/elara-math/tree/main/examples).

## Usage

To use `elara-math` for your own project, simply add it to your project with Cargo:

```sh
cargo add elara-math elara_log
```

Then in your code, just import the library:

```rust
use elara_log::prelude::*; // this is required
use elara_math::prelude::*; // load prelude

fn main() {
    // Initialize elara-math's logging
    // library first
    Logger::new().init().unwrap();

    // rest of your code
    // ...
}
```

The library's prelude is designed for user-friendliness and contains a variety of modules pre-loaded. For those who want finer-grained control, you can individually import the modules you need.

## Developing

To develop `elara-math`, first clone the repository:

```sh
git clone https://github.com/elaraproject/elara-math
git submodule update --init --recursive
```

Then, copy over the pre-commit githook:

```sh
cp .githooks/pre-commit .git/hooks/pre-commit && chmod a+x .git/hooks/pre-commit
```

You should then be all set to start making changes!
