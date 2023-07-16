use matrixmultiply;

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
/// where A dims = M x N
/// and B dims = N x K
/// and C dims M x K
pub fn dgemm(m: usize, n: usize, k: usize, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    // TODO: optimize this or replace with actual BLAS
    let mut c = vec![0.0; m * k];
    unsafe {
   		matrixmultiply::dgemm(m, n, k, 1.0, a.as_ptr(), n as isize, 1, b.as_ptr(), k as isize, 1, 0.0, c.as_mut_ptr(), k as isize, 1);
   	}
   	c
}
