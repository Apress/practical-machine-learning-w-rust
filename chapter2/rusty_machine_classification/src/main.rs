use std::io;
use std::vec::Vec;
use std::error::Error;

use rusty_machine as rm;
use rm::linalg::Matrix;
use rm::linalg::Vector;
use rm::learning::knn::KNNClassifier;
use rusty_machine::learning::knn::{KDTree, BallTree, BruteForce};
use rm::learning::SupModel;
use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use ml_utils;
use ml_utils::datasets::Flower;
use ml_utils::sup_metrics::accuracy;

fn main() -> Result<(), Box<Error>> {
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
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the labels.
    let flower_x_train: Vec<f64> = train_data.iter().flat_map(|r| {
        let features = r.into_feature_vector();
        let features: Vec<f64> = features.iter().map(|&x| x as f64).collect();
        features
    }).collect();
    let flower_y_train: Vec<usize> = train_data.iter().map(
        |r| r.into_int_labels() as usize).collect();

    let flower_x_test: Vec<f64> = test_data.iter().flat_map(|r| {
        let features = r.into_feature_vector();
        let features: Vec<f64> = features.iter().map(|&x| x as f64).collect();
        features
    }).collect();
    let flower_y_test: Vec<u32> = test_data.iter().map(|r| r.into_int_labels() as u32).collect();

    // COnvert the data into matrices for rusty machine
    let flower_x_train = Matrix::new(train_size, 4, flower_x_train);
    let flower_y_train = Vector::new(flower_y_train);
    let flower_x_test = Matrix::new(test_size, 4, flower_x_test);

    // train the classifier to search 2 nearest. this is the same as kdtree
    let mut knn = KNNClassifier::new(2);
    println!("{:?}", knn);

    // train the classifier
    knn.train(&flower_x_train, &flower_y_train).unwrap();

    // predict new points
    let preds = knn.predict(&flower_x_test).unwrap();
    let preds: Vec<u32> = preds.data().iter().map(|&x| x as u32).collect();
    println!("default is binary tree");
    println!("accuracy {:?}", accuracy(preds.as_slice(), &flower_y_test));

    // Ball tree is good when the number of dimensions are huge.
    let mut knn = KNNClassifier::new_specified(2, BallTree::new(30));
    println!("{:?}", knn);

    // train the classifier
    knn.train(&flower_x_train, &flower_y_train).unwrap();

    // predict new points
    let preds = knn.predict(&flower_x_test).unwrap();
    let preds: Vec<u32> = preds.data().iter().map(|&x| x as u32).collect();
    println!("accuracy for ball tree {:?}", accuracy(preds.as_slice(), &flower_y_test));

    // The k-d tree is a binary tree in which every leaf node is a k-dimensional point
    let mut knn = KNNClassifier::new_specified(2, KDTree::default());
    println!("{:?}", knn);

    // train the classifier
    knn.train(&flower_x_train, &flower_y_train).unwrap();

    // predict new points
    let preds = knn.predict(&flower_x_test).unwrap();
    let preds: Vec<u32> = preds.data().iter().map(|&x| x as u32).collect();
    println!("accuracy for kdtree tree {:?}", accuracy(preds.as_slice(), &flower_y_test));

    // The k-d tree is a binary tree in which every leaf node is a k-dimensional point
    let mut knn = KNNClassifier::new_specified(2, KDTree::default());
    println!("{:?}", knn);

    // train the classifier
    knn.train(&flower_x_train, &flower_y_train).unwrap();

    // predict new points
    let preds = knn.predict(&flower_x_test).unwrap();
    let preds: Vec<u32> = preds.data().iter().map(|&x| x as u32).collect();
    println!("accuracy for ball tree {:?}", accuracy(preds.as_slice(), &flower_y_test));

    // Brute force means all the nearest neighbors are looked into
    let mut knn = KNNClassifier::new_specified(2, BruteForce::default());
    println!("{:?}", knn);

    // train the classifier
    knn.train(&flower_x_train, &flower_y_train).unwrap();

    // predict new points
    let preds = knn.predict(&flower_x_test).unwrap();
    let preds: Vec<u32> = preds.data().iter().map(|&x| x as u32).collect();
    println!("accuracy for brute force {:?}", accuracy(preds.as_slice(), &flower_y_test));


    Ok(())
}
