/// Performs the operation A.T -> B
/// where A dims = M x N
pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut b = vec![0.0; n * m];

    for i in 0..n {
        for j in 0..m {
            b[(m * i) + j] = a[(n * j) + i]
        }
    }
    b
}

/// Matrix multiplication
/// Performs AB -> C
/// where A dims = N x M
/// and B dims = M x P
/// and C dims N x P
pub fn dgemm(n: usize, m: usize, p: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    // TODO: optimize this or replace with actual BLAS
    let mut c = vec![0.0; n * p];
    for i in 0..n {
        for j in 0..p {
            for k in 0..m {
                c[(p * i) + j] += a[(m * i) + k] * b[(p * k) + j];
            }
        }
    }
    c
}
