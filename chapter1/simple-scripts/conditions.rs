fn main() {
	let place = "himalayas";

	let weather = if place == "himalayas" {
		"cold"
	} else {
		"hot"
	};
	println!("{:?}", weather);
}