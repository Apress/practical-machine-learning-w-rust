// ownership1.rs

fn main() {
    let lang = "rust";
    let rust1 = add_version(&lang);
    println!("{:?}", rust1);
}

fn add_version(s: &str) -> String {
    s.to_string() + " 2018."
}
