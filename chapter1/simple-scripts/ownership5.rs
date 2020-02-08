// $ ./ownership
// "rust 2018."
// "rust lang."

fn main() {
    let lang = "rust"; // change done here
    let rust1 = add_version(&lang); // change done here
    println!("{:?}", rust1);
    let rust2 = add_lang(&lang); // change done here
    println!("{:?}", rust2);
}

fn add_version(s: &str) -> String {
    s.to_string() + " 2018."
}

fn add_lang(s: &str) -> String {
    s.to_string() + " lang."
}