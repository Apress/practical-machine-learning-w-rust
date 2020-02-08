use faster::*;
use rblas::Dot;
// use lapack::*;

fn main() {
    let lots_of_3s = (&[-123.456f32; 128][..]).simd_iter(f32s(0.0))
        .simd_map(|v| {
            f32s(9.0) * v.abs().sqrt().rsqrt().ceil().sqrt() - f32s(4.0) - f32s(2.0)
        })
        .scalar_collect();
    println!("{:?}", lots_of_3s);

    // making a parallel operation
    let my_vector: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let power_of_3 = (&my_vector[..]).simd_iter(f32s(0.0))
        .simd_map(|v| {
            v * v * v
        })
        .scalar_collect();
    println!("{:?}", power_of_3);

    // taking the sum
    let reduced = (&power_of_3[..]).simd_iter(f32s(0.0))
        .simd_reduce(f32s(0.0), |a, v| a + v ).sum();
    println!("{:?}", reduced);

    let x = vec![1.0, -2.0, 3.0, 4.0];
    let y = [1.0, 1.0, 1.0, 1.0, 7.0];

    let d = Dot::dot(&x, &y[..x.len()]);
    println!("dot product {:?}", d);

    // let n = 3;
    // let mut a = vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
    // let mut w = vec![0.0; n as usize];
    // let mut work = vec![0.0; 4 * n as usize];
    // let lwork = 4 * n;
    // let mut info = 0;

    // unsafe {
    //     dsyev(b'V', b'U', n, &mut a, n, &mut w, &mut work, lwork, &mut info);
    // }

    // assert!(info == 0);
    // for (one, another) in w.iter().zip(&[2.0, 2.0, 5.0]) {
    //     assert!((one - another).abs() < 1e-14);
    // }

}
