#![allow(non_snake_case)]

use std::error::Error;
use std::result::Result;
use std::vec::Vec;

use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;
use transpose;

use tensorflow as tf;
use tf::{Graph, Tensor, DataType};
use tf::{Session, SessionOptions, SessionRunArgs};

use ml_utils;
use ml_utils::datasets::get_boston_records_from_file;
use ml_utils::sup_metrics::r_squared_score;

use random;
use random::Source;
use std::path::Path;
use std::process::exit;
use tensorflow::Code;
use tensorflow::Status;

#[cfg_attr(feature="examples_system_alloc", global_allocator)]
#[cfg(feature="examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub fn run() -> Result<(), Box<dyn Error>> {
    // Get all the data
    let filename = "data/housing.csv";
    let mut data = get_boston_records_from_file(&filename);

    // shuffle the data.
    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets.
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the targets.
    let boston_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();

    let boston_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    // println!("{:?}", boston_y_train.len());
    println!("{:?}", boston_x_train.len());

    // Define graph.
    let mut graph = Graph::new();
    let dim = (boston_y_train.len() as u64, 13);
    let test_dim = (boston_y_test.len() as u64, dim.1);
    let X_train = <Tensor<f64>>::new(&[dim.0, dim.1]).with_values(&boston_x_train)?;
    let y_train = <Tensor<f64>>::new(&[dim.0, 1]).with_values(&boston_y_train)?;
    let X_test = <Tensor<f64>>::new(&[test_dim.0, test_dim.1]).with_values(&boston_x_test)?;
    // let y_test = <Tensor<f64>>::new(&[test_dim.0, 1]).with_values(&boston_y_test)?;

    let export_dir = "boston_regression/"; // y = w * x + b
    if !Path::new(export_dir).exists() {
        return Err(Box::new(Status::new_set(Code::NotFound,
                                            &format!("Run the code in the tensorflow notebook in \
                                                      {} and try again.",
                                                     export_dir))
            .unwrap()));
    }

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let session = Session::from_saved_model(&SessionOptions::new(),
                                            &["train", "serve"],
                                            &mut graph,
                                            export_dir)?;
    let op_x = graph.operation_by_name_required("x")?;
    let op_x_test = graph.operation_by_name_required("x_test")?;
    let op_y = graph.operation_by_name_required("y")?;
    let op_train = graph.operation_by_name_required("train")?;
    let op_w = graph.operation_by_name_required("w")?;
    let op_y_preds = graph.operation_by_name_required("y_preds")?;

    Session::new(&SessionOptions::new(), &graph)?;
    let mut args = SessionRunArgs::new();
    args.add_feed(&op_x, 0, &X_train);
    args.add_feed(&op_x_test, 0, &X_test);
    args.add_feed(&op_y, 0, &y_train);
    args.add_target(&op_train);
    let preds_token = args.request_fetch(&op_y_preds, 0);
    for _ in 0..10 {
        session.run(&mut args)?;
    };
    let preds_token_res: Tensor<f64> = args.fetch::<f64>(preds_token)?;
    println!("Now the preds", );
    println!("{:?}", &preds_token_res[..]);
    println!("{:?}", &boston_y_test);
    println!("{:?}", r_squared_score(&preds_token_res[..], &boston_y_test));

    Ok(())
}
