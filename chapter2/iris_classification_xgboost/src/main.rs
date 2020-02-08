extern crate serde;
// This lets us write `#[derive(Deserialize)]`.
#[macro_use]
extern crate serde_derive;

use std::io;
use std::process;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use xgboost;
use xgboost::{parameters, DMatrix, Booster};

use ml_utils::datasets::Flower;

fn main() {
    if let Err(err) = read_csv() {
        println!("{}", err);
        process::exit(1);
    }
}

fn read_csv() -> Result<(), Box<Error>> {
    // Get all the data
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: Flower = result?;
        data.push(r); // data contains all the records
    }

    // shuffle the data.
    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets.
    let test_size: f32 = 0.2;
    let test_size: f32 = data.len() as f32 * test_size;
    let test_size = test_size.round() as usize;
    // we are keeping the val size to be the same as test_size.
    // this can be changed if required
    let val_size  = test_size.clone();

    let (test_data, train_and_val_data) = data.split_at(test_size);
    let (val_data, train_data) = train_and_val_data.split_at(val_size);
    let train_size = train_data.len();
    let test_size = test_data.len();
    let val_size = val_data.len();

    // differentiate the features and the labels.
    let flower_x_train: Vec<f32> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_train: Vec<f32> = train_data.iter().map(|r| r.into_labels()).collect();

    let flower_x_test: Vec<f32> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_test: Vec<f32> = test_data.iter().map(|r| r.into_labels()).collect();

    let flower_x_val: Vec<f32> = val_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_val: Vec<f32> = val_data.iter().map(|r| r.into_labels()).collect();

    // convert training data into XGBoost's matrix format
    let mut dtrain = DMatrix::from_dense(&flower_x_train, train_size).unwrap();

    // set ground truth labels for the training matrix
    dtrain.set_labels(&flower_y_train).unwrap();

    // test matrix with 1 row
    let mut dtest = DMatrix::from_dense(&flower_x_test, test_size).unwrap();
    dtest.set_labels(&flower_y_test).unwrap();

    // validation matrix with 1 row
    let mut dval = DMatrix::from_dense(&flower_x_val, val_size).unwrap();
    dval.set_labels(&flower_y_val).unwrap();

    // configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::MultiSoftmax(3))
        .build().unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build().unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build().unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // overall configuration for training/evaluation
    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)                         // dataset to train with
        .boost_rounds(2)                         // number of training iterations
        .booster_params(booster_params)          // model parameters
        .evaluation_sets(Some(evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build().unwrap();

    // train model, and print evaluation data
    let booster = Booster::train(&params).unwrap();

    // get predictions
    let preds = booster.predict(&dval).unwrap();
    println!("preds: {:?}", preds);

    // true values
    let labels = dval.get_labels().unwrap();
    println!("{:?}", labels);

    // find the accuracy
    let mut hits = 0;
    let mut correct_hits = 0;
    for (predicted, actual) in preds.iter().zip(labels.iter()) {
        if predicted == actual {
            correct_hits += 1;
        }
        hits += 1;
    }
    assert_eq!(hits, preds.len());
    println!("accuracy={} ({}/{} correct)", correct_hits as f32 / hits as f32, correct_hits, preds.len());

    Ok(())
}