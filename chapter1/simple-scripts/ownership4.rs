// $ ./ownership4
// "rust 2019!!"
// "rust 2019!! lang."

fn main() {
    let mut lang = String::from("rust");
    let rust1 = add_version(&mut lang);
    println!("{:?}", rust1);
    let rust2 = add_lang(&mut lang);
    println!("{:?}", rust2);
}

fn add_version(s: &mut String) -> String {
    s.push_str(" 2019!!");
    s.to_string()
}

fn add_lang(s: &mut String) -> String {
    s.push_str(" lang.");
    s.to_string()
}