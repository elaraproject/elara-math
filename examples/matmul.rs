use elara_math::array;

fn main() {
    let t1 = array![[1.0, 2.0], [3.0, 4.0]];
    let t2 = array![[5.0, 6.0], [7.0, 8.0]];
    println!("{:?}", t1);
    println!("{:?}", t2);
    // Expected: [[19, 22], [43, 50]]
    println!("{:?}", t1.matmul(&t2));
}
