use std::io;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use tch;
use tch::{nn, kind, Kind, Tensor, no_grad, vision, Device};
use tch::{nn::Module, nn::OptimizerConfig};

use ml_utils;
use ml_utils::datasets::Flower;

static FEATURE_DIM: i64 = 4;
static HIDDEN_NODES: i64 = 10;
static LABELS: i64 = 3;

#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let fc1 = nn::Linear::new(vs, FEATURE_DIM, HIDDEN_NODES, Default::default());
        let fc2 = nn::Linear::new(vs, HIDDEN_NODES, LABELS, Default::default());
        Net { fc1, fc2 }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).relu().apply(&self.fc2)
    }
}

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
    let test_size: f64 = 0.5;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;

    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();
    assert_eq!(train_size, test_size);

    // differentiate the features and the labels.
    // torch needs vectors in f64
    let flower_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).map(|x| x as f64).collect();
    let flower_y_train: Vec<f64> = train_data.iter().map(|r| r.into_labels()).map(|x| x as f64).collect();

    let flower_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).map(|x| x as f64).collect();
    let flower_y_test: Vec<f64> = test_data.iter().map(|r| r.into_labels()).map(|x| x as f64).collect();

    let flower_x_train = Tensor::float_vec(flower_x_train.as_slice());
    let flower_y_train = Tensor::float_vec(flower_y_train.as_slice()).to_kind(Kind::Int64);
    let flower_x_test = Tensor::float_vec(flower_x_test.as_slice());
    let flower_y_test = Tensor::float_vec(flower_y_test.as_slice()).to_kind(Kind::Int64);

    // print shape of all the data.
    println!("Training data shape {:?}", flower_x_train.size());
    println!("Training flower_y_train data shape {:?}", flower_y_train.size());

    // reshaping examples
    // one way to reshape is using unsqueeze
    //let flower_x_train1 = flower_x_train.unsqueeze(0); // Training data shape [1, 360]
    //println!("Training data shape {:?}", flower_x_train1.size());
    let train_size = train_size as i64;
    let test_size = test_size as i64;
    let flower_x_train = flower_x_train.view(&[train_size, FEATURE_DIM]);
    let flower_x_test = flower_x_test.view(&[test_size, FEATURE_DIM]);
    let flower_y_train = flower_y_train.view(&[train_size]);
    let flower_y_test = flower_y_test.view(&[test_size]);

    // working on a linear neural network with SGD
    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());
    let opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..200 {
            let loss = net
                .forward(&flower_x_train)
                .cross_entropy_for_logits(&flower_y_train);
            opt.backward_step(&loss);
            let test_accuracy = net
                .forward(&flower_x_test)
                .accuracy_for_logits(&flower_y_test);
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                100. * f64::from(&test_accuracy),
            );
    };

    Ok(())
}
