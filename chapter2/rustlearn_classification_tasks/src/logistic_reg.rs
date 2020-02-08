use std::io;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use rustlearn::prelude::*;
use rustlearn::linear_models::sgdclassifier::Hyperparameters as logistic_regression;
use rustlearn::metrics::{accuracy_score, roc_auc_score};

use ml_utils;
use ml_utils::sup_metrics::{accuracy, logloss_score};
use ml_utils::datasets::Flower;

pub fn run() -> Result<(), Box<dyn Error>> {
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
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the labels.
    let flower_x_train: Vec<f32> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_train: Vec<f32> = train_data.iter().map(|r| r.into_labels()).collect();

    let flower_x_test: Vec<f32> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_test: Vec<f32> = test_data.iter().map(|r| r.into_labels()).collect();

    // Since rustlearn works with arrays we need to convert the vectors to a dense matrix or a sparse matrix
    let mut flower_x_train = Array::from(flower_x_train); // as opposed to rusty machine, all floats here are f32 reference : https://github.com/maciejkula/rustlearn/blob/7daf692fe504966aa84d920321b884afe19caa79/src/array/dense.rs#L129
    flower_x_train.reshape(train_size, 4);

    let flower_y_train = Array::from(flower_y_train);

    let mut flower_x_test = Array::from(flower_x_test);
    flower_x_test.reshape(test_size, 4);

    let flower_y_test = Array::from(flower_y_test);

    // working with Stochastic Gradient descent.
    // uses adaptive per parameter learning rate Adagrad
    let mut model = logistic_regression::new(4)
        .learning_rate(1.0)
        .l2_penalty(0.5)
        .l1_penalty(0.0)
        .one_vs_rest();
    let num_epochs = 100;

    for _ in 0..num_epochs {
        model.fit(&flower_x_train, &flower_y_train).unwrap();
    }

    let prediction = model.predict(&flower_x_test).unwrap();
    let acc1 = accuracy_score(&flower_y_test, &prediction);
    // let acc2 = accuracy(flower_y_test.data()[..], prediction.data()[..]);
    println!("Logistic Regression: accuracy: {:?}", acc1);
    // println!("Logistic Regression: accuracy: {:?}", acc2);

    Ok(())
}