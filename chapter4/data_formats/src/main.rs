#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate serde_xml_rs;

use std::vec::Vec;
use std::process::exit;
use std::env::args;

mod jsonreading;
mod xmlreading;
mod csvreading;

fn main() {
    let args: Vec<String> = args().collect();
    let model = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };
    let res = match model {
        None => {println!("nothing", ); Ok(())},
        Some("json") => jsonreading::run(),
        Some("xml") => xmlreading::run(),
        Some(_) => csvreading::run(),
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