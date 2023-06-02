use std::time::{SystemTime, UNIX_EPOCH};
// Re-export common constants
pub use std::f64::consts::*;

/// Pseudorandom generator based on
/// <https://users.rust-lang.org/t/random-number-without-using-the-external-crate/17260/11>
pub fn rand() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
}

/// Generates a random float
pub fn randf() -> f64 {
    (rand() as f64) / (<u32>::max_value() as f64)
}

/// Generates a random int between 2 numbers
pub fn randint(a: i32, b: i32) -> i32 {
    let m = (b - a + 1) as u32;
    a + (rand() % m) as i32
}

/// Finds the maximum of 2 values
pub fn max<T>(a: T, b: T) -> T
where
    T: PartialOrd,
{
    if a >= b {
        a
    } else {
        b
    }
}
