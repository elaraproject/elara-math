// source: https://github.com/TheAlgorithms/Rust/blob/master/src/math/simpson_integration.rs
pub fn simpson_integration<F: Fn(f64) -> f64>(
    start: f64,
    end: f64,
    steps: u64,
    function: F,
) -> f64 {
    let mut result = function(start) + function(end);
    let step = (end - start) / steps as f64;
    for i in 1..steps {
        let x = start + step * i as f64;
        match i % 2 {
            0 => result += function(x) * 2.0,
            1 => result += function(x) * 4.0,
            _ => unreachable!(),
        }
    }
    result *= step / 3.0;
    result
}
