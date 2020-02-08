use std::result::Result;
use std::vec::Vec;
use std::error::Error;
use std::fs::File;
use std::env;
use std::ffi::OsString;

use csv;
use csv::ReaderBuilder;

#[derive(Debug)]
#[derive(Deserialize)]
struct IrisRecord {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

/// Returns the first positional argument sent to this process. If there are no
/// positional arguments, then this returns an error.
fn get_first_arg() -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(2) {
        None => Err(From::from("expected 2 arguments, but got none")),
        Some(file_path) => Ok(file_path),
    }
}
pub fn run() -> Result<(), Box<dyn Error>> {
    let file_path = get_first_arg()?;
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);
    let mut iris_matrix: Vec<IrisRecord> = vec![];

    for result in rdr.deserialize() {
        let record: IrisRecord = result?;
        iris_matrix.push(record);
    }
    println!("{:#?}", iris_matrix);

    Ok(())
}