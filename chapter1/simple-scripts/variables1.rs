#![feature(core_intrinsics)]

fn print_type_of<T>(_: &T) {
    println!("{}", unsafe { std::intrinsics::type_name::<T>() });
}

fn main() {
    let x = "learning rust";
    let y = 6;
    let z = 3.14;

    println!("{}", x);
    println!("type of x:");
    print_type_of(&x);
    println!("type of y:");
    print_type_of(&y);
    println!("type of z:");
    print_type_of(&z);

}
