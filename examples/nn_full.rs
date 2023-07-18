use elara_math::prelude::*;

fn main() {
    let x = Tensor::arange(-5..5, [10, 1]);
    let func = &x * -5.0;
    let y = &func + &(Tensor::rand([10, 1]) * 0.4);
    println!("Train X:\n{:?}", x);
    println!("Train Y:\n{:?}", y);

    let mut model = Model::new();
    model.add_layer(Linear::new(1, 10, Activations::ReLU));
    model.add_layer(Linear::new(10, 10, Activations::ReLU));
    model.add_layer(Linear::new(10, 1, Activations::None));

    model.fit(&x, &y, 1000, 0.001, false);

    let x_pred = scalar!(1.0);
    let y_pred = model.predict(&x_pred);
    println!("Predict result (should be -5): {:?}", y_pred);
}