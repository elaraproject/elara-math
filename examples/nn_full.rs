use elara_math::prelude::*;
use elara_log::prelude::*;

fn main() {
    Logger::new().init().unwrap();
    let x = Tensor::linspace(-5.0, 5.0, 50);
    let func = &x * -5.0;
    let y = &func + &(Tensor::rand([50, 1]) * 0.4);

    let mut model = Model::new();
    model.add_layer(Linear::new(1, 10, Activations::None));
    model.add_layer(Linear::new(10, 10, Activations::None));
    model.add_layer(Linear::new(10, 1, Activations::None));

    model.compile(Optimizers::BGD);
    model.fit(&x, &y, 2000, 0.001, true);

    let x_pred = scalar!(1.0);
    let y_pred = model.predict(&x_pred);
    println!("Predict result (should be -5): {:?}", y_pred);
}