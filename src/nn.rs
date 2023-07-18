use crate::Tensor;
use crate::mse;
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
    pub activation: Activations
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activations) -> Linear {
        let weights = Array2::random((input_dim, output_dim), Uniform::new(0.0, 1.0));
        let biases = Array2::zeros((1, output_dim));
        Linear {
            weights: Tensor::new(weights),
            biases: Tensor::new(biases),
            activation
        }
    }
}

impl Layer for Linear {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut p: Vec<&Tensor> = vec![&self.weights, &self.biases];
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

pub struct Model {
    pub layers: Vec<Linear>,
}

impl Model {
    pub fn new() -> Model {
        Model { layers: vec![] }
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

    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: usize, lr: f64, debug: bool) {
        for epoch in 0..(epochs + 1) {
            let out = self.forward(x);
            let loss = mse(&out, y);
            if debug {
                println!("Epoch {}, loss {:?}", epoch, loss);
            }
            loss.backward();
            self.update(lr);
        }
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
}