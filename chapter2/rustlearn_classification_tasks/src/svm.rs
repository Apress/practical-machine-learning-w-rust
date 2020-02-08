use std::io;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use rustlearn::prelude::*;
use rustlearn::svm::libsvm::svc::{Hyperparameters as libsvm_svc, KernelType};
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

    // Working with svms
    let svm_linear_model = libsvm_svc::new(4, KernelType::Linear, 3)
        .C(0.3)
        .build();
    let svm_poly_model = libsvm_svc::new(4, KernelType::Polynomial, 3)
        .C(0.3)
        .build();
    let svm_rbf_model = libsvm_svc::new(4, KernelType::RBF, 3)
        .C(0.3)
        .build();
    let svm_sigmoid_model = libsvm_svc::new(4, KernelType::Sigmoid, 3)
        .C(0.3)
        .build();
    let svm_kernel_types = ["linear", "polynomial", "rbf", "sigmoid"];
    let mut svm_model_types = [svm_linear_model, svm_poly_model, svm_rbf_model, svm_sigmoid_model];
    for (kernel_type, svm_model) in svm_kernel_types.iter().zip(svm_model_types.iter_mut()) {
        svm_model.fit(&flower_x_train, &flower_y_train).unwrap();

        let prediction = svm_model.predict(&flower_x_test).unwrap();
        let acc = accuracy_score(&flower_y_test, &prediction);
        println!("Lib svm {kernel}: accuracy: {accuracy}", accuracy=acc, kernel=kernel_type);
    };

    let preds = vec![1., 0.0001, 0.908047338626, 0.0199900075962, 0.904058545833, 0.321508119045, 0.657086320195];
    let actuals = vec![1., 0., 0., 1., 1., 0., 0.];
    println!("logloss score: {:?}", logloss_score(&actuals, &preds, 1e-15));
    println!("roc auc scores: {:?}", roc_auc_score(&Array::from(actuals), &Array::from(preds))?);

    Ok(())
}