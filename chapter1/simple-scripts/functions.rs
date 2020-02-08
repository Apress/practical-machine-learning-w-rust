fn main() {
	println!("{:?}", square_of(-5));
}

fn square_of(x: i32) -> i32 {
	println!("x = {:?}", x);
	x.pow(2)
}