use elara_math::{tensor, Tensor, exp};

fn main() {
    // let train_data: Tensor<f64, 2> = tensor!([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]);
    // let train_labels: Tensor<f64, 2> = tensor!([[0.0, 1.0, 1.0, 0.0]]).transpose();
     #[rustfmt::skip]
    let train_data = Tensor::new(&[
        0.0, 0.0, 1.0, 
        1.0, 1.0, 1.0, 
        1.0, 0.0, 1.0, 
        0.0, 1.0, 1.0], [4, 3]);
    let train_labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0], [1, 4]).t();
    let mut weights = Tensor::random([3, 1]) * 2.0 - 1.0;
    let test = train_data.clone();
    let test_labels = train_labels.clone();
    println!("{:?}", weights);
    for i in 0..10000 {
        // println!("Training epoch {}", i)
        let n = test.dot(weights);
        let output = 1.0 / (1.0 + exp(-n));
        let m = (&test_labels - output) * output * (1.0 - output);
        let k = train_data.t().dot(m);
        let new_weights = weights.clone();
        weights = weights.clone() + k;
        /* 
        for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
        
         */
    }
    let pred_data: Tensor<f64, 2> = tensor!([1.0, 0.0, 0.0]);
    let pred = 1.0 / (1.0 + exp(pred_data.dot(weights)));
    println!("Prediction: {}", pred);
}
