#![allow(non_snake_case)]

use std::error::Error;
use std::result::Result;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;

use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;
use transpose;
use mnist;
use mnist::{Mnist, MnistBuilder};

use tensorflow as tf;
use tf::expr::{Compiler, Constant};
use tf::{Graph, Tensor, DataType, Shape};
use tf::{Session, SessionOptions, SessionRunArgs};

#[cfg_attr(feature="examples_system_alloc", global_allocator)]
#[cfg(feature="examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub fn run() -> Result<(), Box<dyn Error>> {
    // Get all the data
    let (trn_size, rows, cols) = (10_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    // Get the label of the first digit.
    let first_label = trn_lbl[0];
    println!("The first digit is a {}.", first_label);
    println!("size of training {}.", trn_img.len());

    let trn_img: Vec<f64> = trn_img.iter().map(|&x| x as f64).collect();
    let trn_lbl: Vec<f64> = trn_lbl.iter().map(|&x| x as f64).collect();


    // Define graph.
    let mut graph = Graph::new();
    let X = <Tensor<f64>>::new(&[10_000, 28, 28, 1]).with_values(&trn_img[..])?;
    let y = <Tensor<f64>>::new(&[10_000,]).with_values(&trn_lbl[..])?;
    let z = <Tensor<f64>>::new(&[28, 28, 1, 32]).with_values(&vec![1.; 25088])?;

    let X_const = {
        let mut c = graph.new_operation("Placeholder", "X")?;
        c.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
        // c.set_attr_shape("shape", &Shape::from(Some(vec![Some(28),Some(28),Some(1),Some(32)])))?;
        c.set_attr_shape("shape", &Shape::from(Some(vec![Some(10_000),Some(28),Some(28),Some(1)])))?;
        c.finish()?
    };
    // operation types https://github.com/malmaud/TensorFlow.jl/blob/063511525902bdf84a461035758ef9a73ba4a635/src/ops/op_names.txt
    let max_pool = {
        let mut op = graph.new_operation("MaxPool", "max_pool")?;
        op.add_input(X_const.clone());
        op.set_attr_string("padding", "VALID")?;
        op.set_attr_int_list("strides", &[1,2,2,1])?;
        op.set_attr_int_list("ksize", &[1,2,2,1])?;
        op.finish()?
    };

    // Run graph.
    let session = Session::new(&SessionOptions::new(), &graph)?;
    let mut args = SessionRunArgs::new();
    args.add_feed(&X_const, 0, &X);
    let max_pool_token = args.request_fetch(&max_pool, 0);
    session.run(&mut args)?;
    let max_pool_token_res: Tensor<f64> = args.fetch::<f64>(max_pool_token)?;
    println!("Now the max_pool", );
    println!("{:?}", &max_pool_token_res[..]);

    Ok(())
}
