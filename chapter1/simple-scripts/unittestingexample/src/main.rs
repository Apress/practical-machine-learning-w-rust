fn main() {
    let lang = "rust";
    let rust1 = add_version(&lang);
    println!("{:?}", rust1);
}

fn add_version(s: &str) -> String {
    s.to_string() + " 2018."
}

#[test]
fn test_add_version() {
    assert_eq!(add_version("abcd"), String::from("abcd 2018."));
}