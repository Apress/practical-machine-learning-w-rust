use ndarray;
use ndarray::prelude::*;
use ndarray::stack;

fn main() {
    let a1 = arr2(&[[0., 1., 2.],
                    [3., 4., 5.]]);
    println!("{:?}", a1);
    println!("------------------------", );

    let a2 = Array::from_shape_vec((2, 3).strides((3, 1)),
        vec![0., 1., 2., 3., 4., 5.]).unwrap();
    assert!(a1 == a2);

    let a_T = a1.t();
    println!("transposed matrix:");
    println!("{:?}", a_T);
    println!("------------------------", );

    let a_mm = a1.dot(&a_T);
    println!("dot product:");
    println!("{:?}", a_mm);
    println!("{:?}", a_mm.shape()); // output [2, 2]
    println!("------------------------", );
}
