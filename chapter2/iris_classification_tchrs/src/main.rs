extern crate serde;
// This lets us write `#[derive(Deserialize)]`.
#[macro_use]
extern crate serde_derive;

use std::vec::Vec;
use std::process::exit;
use std::env::args;

mod simple_nn;
mod linear_with_sgd;

fn main() {
    let args: Vec<String> = args().collect();
    let model = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };
    let res = match model {
        None => {println!("Run cargo run [nn|linear sdg] to get outputs", ); Ok(())},
        Some("nn") => simple_nn::run(),
        Some(_) => linear_with_sgd::run(),
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