// ownership3.rs

fn main() {
    let lang = String::from("rust");
    let rust1 = add_version(&lang);
    println!("{:?}", rust1);
    let rust2 = add_lang(&lang);
    println!("{:?}", rust2);
}

fn add_version(s: &String) -> String {
    s.push_str(" 2019!!");
    s.to_string()
}

fn add_lang(s: &String) -> String {
    s.push_str(" lang.");
    s.to_string()
}
