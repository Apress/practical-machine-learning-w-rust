use std::io;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use rustlearn::prelude::*;
use rustlearn::ensemble::random_forest::Hyperparameters as randomforest;
use rustlearn::trees::decision_tree;
use rustlearn::metrics::{accuracy_score, roc_auc_score};

use ml_utils;
use ml_utils::sup_metrics::{accuracy, logloss_score};
use ml_utils::datasets::Flower;

pub fn run() -> Result<(), Box<Error>> {
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

    // create a decision tree model
    let mut decision_tree_model = decision_tree::Hyperparameters::new(flower_x_train.cols())
        .one_vs_rest();
    decision_tree_model.fit(&flower_x_train, &flower_y_train).unwrap();

    let prediction = decision_tree_model.predict(&flower_x_test).unwrap();
    let acc = accuracy_score(&flower_y_test, &prediction);
    println!("DecisionTree model accuracy: {:?}", acc);

    

    // create a random forest model
    let mut tree_params = decision_tree::Hyperparameters::new(flower_x_train.cols());
    tree_params.min_samples_split(10)
        .max_features(4);

    let mut random_forest_model = randomforest::new(tree_params, 10).one_vs_rest();

    random_forest_model.fit(&flower_x_train, &flower_y_train).unwrap();

    // Optionally serialize and deserialize the model

    // let encoded = bincode::rustc_serialize::encode(&model,
    //                                               bincode::SizeLimit::Infinite).unwrap();
    // let decoded: OneVsRestWrapper<RandomForest> = bincode::rustc_serialize::decode(&encoded).unwrap();

    let prediction = random_forest_model.predict(&flower_x_test).unwrap();
    let acc = accuracy_score(&flower_y_test, &prediction);
    println!("Random Forest: accuracy: {:?}", acc);

    Ok(())
}