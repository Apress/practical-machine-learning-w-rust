use std::error::Error;
use std::result::Result;
use tensorflow as tf;
use tf::expr::{Variable, Compiler};
use tf::{Graph, Tensor};
use tf::{Session, SessionOptions, SessionRunArgs};

#[cfg_attr(feature="examples_system_alloc", global_allocator)]
#[cfg(feature="examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub fn run() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let (x_node, y_node, z_node) = {
        let mut compiler = Compiler::new(&mut g);
        let x_expr = <Variable<f32>>::new_expr(&vec![1], "x");
        let y_expr = <Variable<f32>>::new_expr(&vec![1], "y");
        let y_node = compiler.compile(y_expr.clone())?;
        let x_node = compiler.compile(x_expr.clone())?;

        // let z = x * x * y + y + 2;
        let z_node = compiler.compile(
            x_expr.clone() * x_expr.clone() * y_expr.clone()
            + y_expr.clone() + 2.0f32)?;
        (x_node, y_node, z_node)
    };

    let options = SessionOptions::new();
    let mut session = Session::new(&options, &g)?;

    // Evaluate the graph.
    let x = <Tensor<f32>>::new(&[1]).with_values(&[3.0_f32]).unwrap();
    let y = <Tensor<f32>>::new(&[1]).with_values(&[4.0_f32]).unwrap();
    let mut step = SessionRunArgs::new();
    step.add_feed(&x_node, 0, &x);
    step.add_feed(&y_node, 0, &y);
    let output_token = step.request_fetch(&z_node, 0);
    session.run(&mut step).unwrap();

    // Check our results.
    let output_tensor = step.fetch::<f32>(output_token)?;
    println!("working with variables: z = x * x * y + y + 2 => {:?}", output_tensor[0]);
    session.close()?;

    Ok(())
}