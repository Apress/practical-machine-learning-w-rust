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
    // // Get all the data
    // // let (trn_size, rows, cols) = (10_000, 28, 28);

    // // // Deconstruct the returned Mnist struct.
    // // let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
    // //     .label_format_digit()
    // //     .training_set_length(trn_size)
    // //     .validation_set_length(10_000)
    // //     .test_set_length(10_000)
    // //     .finalize();
    // // // Get the label of the first digit.
    // // let first_label = trn_lbl[0];
    // // println!("The first digit is a {}.", first_label);
    // // println!("size of training {}.", trn_img.len());

    // // let trn_img: Vec<f64> = trn_img.iter().map(|&x| x as f64).collect();
    // // let trn_lbl: Vec<f64> = trn_lbl.iter().map(|&x| x as f64).collect();


    // // Define graph.
    // let mut graph = Graph::new();
    // let X = <Tensor<f64>>::new(&[10_000, 28, 28, 1]).with_values(&trn_img[..])?;
    // let y = <Tensor<f64>>::new(&[10_000,]).with_values(&trn_lbl[..])?;
    // let c1 = <Tensor<f64>>::new(&[28, 28, 1, 32]).with_values(&vec![1.; 25088])?;
    // let c2 = <Tensor<f64>>::new(&[28, 28, 32, 64]).with_values(&vec![1.; 1605632])?;
    // let shp = <Tensor<i64>>::new(&[2]).with_values(&[10_000, 256])?;

    // let X_const = {
    //     let mut c = graph.new_operation("Placeholder", "X")?;
    //     c.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
    //     // c.set_attr_shape("shape", &Shape::from(Some(vec![Some(28),Some(28),Some(1),Some(32)])))?;
    //     c.set_attr_shape("shape", &Shape::from(Some(vec![Some(10_000),Some(28),Some(28),Some(1)])))?;
    //     c.finish()?
    // };
    // // operation types https://github.com/malmaud/TensorFlow.jl/blob/063511525902bdf84a461035758ef9a73ba4a635/src/ops/op_names.txt
    // let filter = {
    //     let mut c = graph.new_operation("Const", "filter")?;
    //     c.set_attr_tensor("value", c1.clone())?;
    //     c.set_attr_type("dtype", DataType::Double)?;
    //     c.finish()?
    // };
    // let conv1 = {
    //     let mut op = graph.new_operation("Conv2D", "conv")?;
    //     // op.add_input(y_const);
    //     op.add_input(X_const.clone());
    //     op.add_input(filter);
    //     op.set_attr_string("padding", "SAME")?;
    //     op.set_attr_int_list("strides", &[1,2,2,1])?;
    //     op.finish()?
    // };
    // let pool1 = {
    //     let mut op = graph.new_operation("MaxPool", "max_pool")?;
    //     op.add_input(conv1);
    //     op.set_attr_string("padding", "VALID")?;
    //     op.set_attr_int_list("strides", &[1,2,2,1])?;
    //     op.set_attr_int_list("ksize", &[1,2,2,1])?;
    //     op.finish()?
    // };

    // // Convolutional Layer #2 and Pooling Layer #2
    // let filter2 = {
    //     let mut c = graph.new_operation("Const", "filter2")?;
    //     c.set_attr_tensor("value", c2)?;
    //     c.set_attr_type("dtype", DataType::Double)?;
    //     c.finish()?
    // };
    // let conv2 = {
    //     let mut op = graph.new_operation("Conv2D", "conv2")?;
    //     op.add_input(pool1);
    //     op.add_input(filter2);
    //     op.set_attr_string("padding", "SAME")?;
    //     op.set_attr_int_list("strides", &[1,2,2,1])?;
    //     op.finish()?
    // };
    // let pool2 = {
    //     let mut op = graph.new_operation("MaxPool", "max_pool2")?;
    //     op.add_input(conv2);
    //     op.set_attr_string("padding", "VALID")?;
    //     op.set_attr_int_list("strides", &[1,2,2,1])?;
    //     op.set_attr_int_list("ksize", &[1,2,2,1])?;
    //     op.finish()?
    // };

    // // Dense layer
    // let shape = {
    //     let mut c = graph.new_operation("Const", "shp")?;
    //     c.set_attr_tensor("value", shp)?;
    //     c.set_attr_type("dtype", DataType::Int64)?;
    //     c.finish()?
    // };
    // let pool2_flat = {
    //     let mut op = graph.new_operation("Reshape", "dense layer")?;
    //     op.add_input(pool2);
    //     op.add_input(shape);
    //     // op.set_attr_shape("shape", &Shape::from(Some(vec![Some(10_000),Some(7 * 7 * 64)])))?;
    //     op.finish()?
    // };
    // let shp3 = <Tensor<i64>>::new(&[2]).with_values(&[1000, 2560])?;
    // let shape3 = {
    //     let mut c = graph.new_operation("Const", "shp3")?;
    //     c.set_attr_tensor("value", shp3)?;
    //     c.set_attr_type("dtype", DataType::Int64)?;
    //     c.finish()?
    // };
    // let dense = {
    //     let mut op = graph.new_operation("Reshape", "dense")?;
    //     op.add_input(pool2_flat);
    //     op.add_input(shape3);
    //     op.finish()?
    // };
    // let activation = {
    //     let mut op = graph.new_operation("Relu", "relu_act")?;
    //     op.add_input(dense);
    //     op.finish()?
    // };
    // // let dropout = { // dropout can be implemented by creating a bernoulli trial vector and then doing a matrix multiplication.
    // //     let mut op = graph.new_operation("Dropout", "dropout")?;
    // //     op.add_input(activation);
    // //     op.finish()?
    // // };
    // // // Define graph.
    // // let mut graph = Graph::new();
    // // let X = <Tensor<f64>>::new(&[10_000, 28, 28, 1]).with_values(&trn_img[..])?;
    // // let y = <Tensor<f64>>::new(&[10_000,]).with_values(&trn_lbl[..])?;
    // // let z = <Tensor<f64>>::new(&[28, 28, 1, 32]).with_values(&vec![1.; 25088])?;

    // // let X_const = {
    // //     let mut c = graph.new_operation("Placeholder", "X")?;
    // //     c.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
    // //     // c.set_attr_shape("shape", &Shape::from(Some(vec![Some(28),Some(28),Some(1),Some(32)])))?;
    // //     c.set_attr_shape("shape", &Shape::from(Some(vec![Some(10_000),Some(28),Some(28),Some(1)])))?;
    // //     c.finish()?
    // // };
    // // let y_const = {
    // //     let mut c = graph.new_operation("Placeholder", "X")?;
    // //     c.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
    // //     // c.set_attr_shape("shape", &Shape::from(Some(vec![Some(28),Some(28),Some(1),Some(32)])))?;
    // //     c.set_attr_shape("shape", &Shape::from(Some(vec![Some(10_000)])))?;
    // //     c.finish()?
    // // };
    // // // operation types https://github.com/malmaud/TensorFlow.jl/blob/063511525902bdf84a461035758ef9a73ba4a635/src/ops/op_names.txt
    // // let filter = {
    // //     let mut c = graph.new_operation("Const", "filter")?;
    // //     c.set_attr_tensor("value", z.clone())?;
    // //     c.set_attr_type("dtype", DataType::Double)?;
    // //     c.finish()?
    // // };
    // // let conv = {
    // //     let mut op = graph.new_operation("Conv2D", "conv")?;
    // //     // op.add_input(y_const);
    // //     op.add_input(X_const.clone());
    // //     op.add_input(filter);
    // //     op.set_attr_string("padding", "VALID")?;
    // //     op.set_attr_int_list("strides", &[1,2,2,1])?;
    // //     op.finish()?
    // // };

    // // let stopped_gradient = {
    // //     let mut nd = g.new_operation("StopGradient", "stopped").unwrap();
    // //     nd.add_input(y_const.clone());
    // //     nd.finish().unwrap()
    // // };
    // // let y_outs = vec![stopped_gradient.into()];
    // // let x_outs = vec![X_const.into()];

    // // Logits layer
    // let shp2 = <Tensor<i64>>::new(&[2]).with_values(&[10, 256000])?;
    // let shape2 = {
    //     let mut c = graph.new_operation("Const", "shp2")?;
    //     c.set_attr_tensor("value", shp2)?;
    //     c.set_attr_type("dtype", DataType::Int64)?;
    //     c.finish()?
    // };
    // let logits = {
    //     let mut op = graph.new_operation("Reshape", "final logits")?;
    //     op.add_input(activation);
    //     op.add_input(shape2);
    //     op.finish()?
    // };

    // // Run graph.
    // let session = Session::new(&SessionOptions::new(), &graph)?;
    // let mut args = SessionRunArgs::new();
    // args.add_feed(&X_const, 0, &X);
    // let theta_token = args.request_fetch(&pool2_flat, 0);
    // session.run(&mut args)?;
    // let theta_token_res: Tensor<f64> = args.fetch::<f64>(theta_token)?;
    // println!("Now the theta", );
    // println!("{:?}", &theta_token_res[..]);

    // // // Run graph.
    // // let session = Session::new(&SessionOptions::new(), &graph)?;
    // // let mut args = SessionRunArgs::new();
    // // args.add_feed(&X_const, 0, &X);
    // // let theta_token = args.request_fetch(&conv, 0);
    // // session.run(&mut args)?;
    // // let theta_token_res: Tensor<f64> = args.fetch::<f64>(theta_token)?;
    // // println!("Now the theta", );
    // // println!("{:?}", &theta_token_res[..]);

    Ok(())
}
