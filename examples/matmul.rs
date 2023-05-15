use elara_math::tensor;

fn main() {
    let t1 = tensor![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    let t2 = tensor![
        [5.0, 6.0],
        [7.0, 8.0]
    ];
    println!("{:?}", t1);
    println!("{:?}", t2);
    // Expected: [[19, 22], [43, 50]]
    println!("{:?}", t1.matmul(&t2));
}
