use crate::Tensor;
use crate::mse;
use elara_log::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub trait Layer {
    fn parameters(&self) -> Vec<&Tensor>;

    fn forward(&self, x: &Tensor) -> Tensor;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

pub struct Linear {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activations,
    input_dim: usize,
    output_dim: usize
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activations) -> Linear {
        let weights = Array2::random((input_dim, output_dim), Uniform::new(0.0, 1.0));
        let biases = Array2::zeros((1, output_dim));
        Linear {
            weights: Tensor::new(weights),
            biases: Tensor::new(biases),
            activation,
            input_dim,
            output_dim
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
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
}

pub enum Activations {
    ReLU,
    Sigmoid,
    None
}

pub enum Optimizers {
    SGD,
    BGD,
    None
}

pub struct Model {
    pub layers: Vec<Linear>,
    pub optimizer: Optimizers
}

impl Model {
    pub fn new() -> Model {
        Model { layers: vec![], optimizer: Optimizers::None }
    }

    pub fn add_layer(&mut self, layer: Linear) { 
        self.layers.push(layer)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.forward(&x);
        }
        x
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    pub fn update(&self, lr: f64) {
        for t in self.parameters().iter() {
            t.update(lr);
        }
    }

    pub fn zero_grad(&self) {
        for t in self.parameters().iter() {
            t.zero_grad();
        }
    }

    pub fn compile(&mut self, optimizer: Optimizers) {
        self.optimizer = optimizer;
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: usize, lr: f64, debug: bool) {
        // Do checks to make sure model is valid
        if self.layers.is_empty() {
            error!("[elara-math] The model does not contain any layers and cannot be trained.")
        }
        match self.optimizer {
            Optimizers::None => { 
                error!("[elara-math] The model was not configured with an optimizer and cannot be trained.")
            },
            _ => {}
        };

        for i in 0..self.layers.len() {
            // Ignore last layer
            if i < self.layers.len() - 1 {
                if self.layers[i].shape().1 != self.layers[i + 1].shape().0 {
                    error!("[elara-math] Layer #{} was configured with an output size of {}, while layer #{} was configured with an input size of {}. This is invalid, both should match.", i, self.layers[i].shape().1, i + 1, self.layers[i + 1].shape().0);
                }
            }
        }

        for epoch in 0..(epochs + 1) {
            match self.optimizer {
                Optimizers::BGD => {
                    let out = self.forward(x);
                    let loss = mse(&out, y);
                    if debug {
                        println!("Epoch {}, loss {:?}", epoch, loss);
                    }
                    loss.backward();
                    self.update(lr);
                    self.zero_grad();
                },
                Optimizers::SGD => {
                    let data = (x.inner_mut(), y.inner_mut());
                    // for (x, y) in (x.data().deref().iter(), y.data().iter()) {}
                    // for (x, y) in zip(x, y) {
                    //     let out = self.forward(x);
                    // }
                },
                _ => unreachable!()
            }
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
}