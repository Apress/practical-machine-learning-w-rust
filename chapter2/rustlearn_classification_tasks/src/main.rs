use std::vec::Vec;
use std::process::exit;
use std::env::args;

mod trees;
mod logistic_reg;
mod svm;
mod binary_class_scores;

fn main() {
    let args: Vec<String> = args().collect();
    let model = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };
    let res = match model {
        None => {println!("nothing", ); Ok(())},
        Some("lr") => logistic_reg::run(),
        Some("svm") => svm::run(),
        Some("bs") => binary_class_scores::run(),
        Some(_) => trees::run(),
    };
    // Putting the main code in another function serves two purposes:
    // 1. We can use the `?` operator.
    // 2. We can call exit safely, which does not run any destructors.
    exit(match res {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}