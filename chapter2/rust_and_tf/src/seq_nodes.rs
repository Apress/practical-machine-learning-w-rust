use std::error::Error;
use std::result::Result;

use tensorflow as tf;
use tf::expr::{Compiler, Constant};
use tf::{Graph, Tensor};
use tf::{Session, SessionOptions, SessionRunArgs};

#[cfg_attr(feature="examples_system_alloc", global_allocator)]
#[cfg(feature="examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub fn run() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let (y_node, z_node) = {
        let mut compiler = Compiler::new(&mut g);
        let w = <Tensor<f32>>::new(&[1]).with_values(&[3.0_f32]).unwrap();
        let w_expr = <Constant<f32>>::new_expr(w);
        let x_expr = w_expr.clone() + 2.0f32;
        let y_expr = x_expr.clone() + 5.0f32;
        let z_expr = x_expr.clone() * 3.0f32;

        let y_node = compiler.compile(y_expr.clone())?;
        let z_node = compiler.compile(z_expr.clone())?;
        (y_node, z_node)
    };

    let options = SessionOptions::new();
    let mut session = Session::new(&options, &g)?;

    // Evaluate the graph.
    let mut step = SessionRunArgs::new();
    let output_token_y = step.request_fetch(&y_node, 0);
    let output_token_z = step.request_fetch(&z_node, 0);
    session.run(&mut step).unwrap();

    // Check our results.
    let output_tensor_y = step.fetch::<f32>(output_token_y)?;
    let output_tensor_z = step.fetch::<f32>(output_token_z)?;
    println!("constant evaluation: w = 3; x = w + 2; y = x + 5; z = x * 3");
    println!("y => {:?}", output_tensor_y[0]);
    println!("z => {:?}", output_tensor_z[0]);
    session.close()?;

    Ok(())
}