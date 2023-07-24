use crate::Tensor;
use crate::mse;
use elara_log::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fmt::Debug;
use std::iter::zip;

const DEBUGGING_GUIDE: &'static str = r#"
This is a helpful debug guide to resolve common issues faced in using elara-math.
If you meet an unexpected error, consider checking the following:

1) Learning rate: Do not set this too high or it will cause divergence,
and do not set this too low or it will cause limited loss reduction by getting
stuck in a local minimum.

2) Training data shapes: Ensure that the training data has the same length as the
test data.

3) Model layer shapes: Ensure that each of the model's layers has the same input
shapes as the prior layer's output shapes.

4) Epoch number: Do not set this too high or it will cause overfitting, or
set it too low or it will cause underfitting
"#;

/// A general trait of a layer of a neural
/// network
pub trait Layer {
    fn parameters(&self) -> Vec<&Tensor>;

    fn forward(&self, x: &Tensor) -> Tensor;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn shape(&self) -> (usize, usize);
}

/// A 2D linearly densely-connected layer
pub struct Linear {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activations,
    input_dim: usize,
    output_dim: usize
}

impl Linear {
    /// Create a new linear layer
    pub fn new(input_dim: usize, output_dim: usize, activation: Activations) -> Linear {
        let weights = Array2::random((input_dim, output_dim), Uniform::new(0.0, 1.0));
        let biases = Array2::random((1, output_dim), Uniform::new(0.0, 0.1));
        Linear {
            weights: Tensor::new(weights),
            biases: Tensor::new(biases),
            activation,
            input_dim,
            output_dim
        }
    }
}

impl Debug for dyn Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer({}, {})", self.shape().0, self.shape().1)
    }
}

impl Layer for Linear {
    fn parameters(&self) -> Vec<&Tensor> {
        let p: Vec<&Tensor> = vec![&self.weights, &self.biases];
        p
    }

    fn forward(&self, train_data: &Tensor) -> Tensor {
        let mut out = &train_data.matmul(&self.weights) + &self.biases;
        out = match &self.activation {
            Activations::ReLU => out.relu(),
            Activations::Sigmoid => out.sigmoid(),
            Activations::None => out
        };
        out
    }

    fn shape(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }
}

/// Common activation functions
pub enum Activations {
    ReLU,
    Sigmoid,
    None
}

/// Common optimizers
pub enum Optimizers {
    SGD,
    BGD,
    None
}

/// A neural network model
/// with a keras-inspired API
pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    pub optimizer: Optimizers
}

impl Model {
    /// Create a new model
    pub fn new() -> Model {
        Model { layers: vec![], optimizer: Optimizers::None }
    }

    /// Add a layer to a model
    pub fn add_layer(&mut self, layer: Linear) { 
        self.layers.push(Box::new(layer))
    }

    /// Compute the forward pass of a model
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.forward(&x);
        }
        x
    }

    /// Get the weights and biases of a model
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    fn update(&self, lr: f64) {
        for t in self.parameters().iter() {
            t.update(lr);
        }
    }

    fn zero_grad(&self) {
        for t in self.parameters().iter() {
            t.zero_grad();
        }
    }

    /// Configure a model with an optimizer
    pub fn compile(&mut self, optimizer: Optimizers) {
        self.optimizer = optimizer;
    }

    /// Train a model
    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: usize, lr: f64, debug: bool) {
        // Do checks to make sure model and input data is valid
        if debug {
            info!("{}", DEBUGGING_GUIDE);
        }

        if self.layers.is_empty() {
            error!("[elara-math] The model does not contain any layers and cannot be trained.")
        }

        match self.optimizer {
            Optimizers::None => { 
                error!("[elara-math] The model was not configured with an optimizer and cannot be trained.")
            },
            _ => {}
        };

        for (idx, (layer, layer_next)) in self.layers.iter().zip(self.layers[1..self.layers.len()].iter()).enumerate() {
            if layer.shape().1 != layer_next.shape().0 {
                error!("[elara-math] Layer #{} was configured with an output size of {}, while layer #{} was configured with an input size of {}. This is invalid, both should match.", idx + 1, layer.shape().1, idx + 2, layer_next.shape().0);
            }
        }

        for epoch in 0..(epochs + 1) {
            match self.optimizer {
                Optimizers::BGD => {
                    let out = self.forward(x);
                    let loss = mse(&out, y);
                    if debug {
                        info!("Epoch {}, loss {:?}", epoch, loss);
                    }
                    loss.backward();
                    self.update(lr);
                    self.zero_grad();
                },
                Optimizers::SGD => {
                    let mut counter = 0;
                    for (x_el, y_el) in zip(x.clone(), y.clone()) {
                        if counter > x.shape().0 {
                            break;
                        }
                        let out = self.forward(&x_el);
                        let loss = mse(&out, &y_el);
                        if debug {
                            info!("Epoch {}, sample {}, loss {:?}", epoch, counter, loss);
                        }
                        loss.backward();
                        self.update(lr);
                        self.zero_grad();
                        counter += 1;
                    }
                },
                _ => unreachable!()
            }
        }
    }

    /// Make predictions from a model
    pub fn predict(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
}