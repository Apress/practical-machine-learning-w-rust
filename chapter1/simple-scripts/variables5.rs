fn main() {
	let x = 5;

	if 4 < 10 {
		let x = 10;
		println!("Inside if x = {:?}", x);
	}
	println!("Outside if x = {:?}", x);
}