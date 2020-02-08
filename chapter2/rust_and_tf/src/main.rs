/// This package is about the different ways you can use tensorflow in rust.
/// Current possible arguments.
/// # Arguments
/// * `` - WIll run the placeholders example.
/// * `seq` - WIll run the sequence of nodes example
/// * `cars` - Will run the example with graph variables
///
/// # Example
/// ```
/// $ cargo run seq
/// ➜  rust_and_tf git:(master) ✗ cargo run seq
///    Finished dev [unoptimized + debuginfo] target(s) in 0.15s
///     Running `target/debug/rust_and_tf seq`
/// 2019-04-07 12:55:41.781908: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
/// 2019-04-07 12:55:41.814069: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 1996260000 Hz
/// 2019-04-07 12:55:41.814902: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x56514fdfded0 executing computations on platform Host. Devices:
/// 2019-04-07 12:55:41.814966: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
/// constant evaluation: w = 3; x = w + 2; y = x + 5; z = x * 3
/// y => 10.0
/// z => 15.0
/// ```

// reference: https://github.com/danieldk/dpar
// https://www.tensorflow.org/tutorials/estimators/cnn

extern crate serde;
// This lets us write `#[derive(Deserialize)]`.
#[macro_use]
extern crate serde_derive;

use std::process::exit;
use std::env::args;

mod graph_with_placeholder;
mod seq_nodes;
mod graph_variables;
mod linear_regression;
mod conv_nets;
mod linear_regression_from_model;
mod conv_nets_maxpooling;


#[cfg_attr(feature="examples_system_alloc", global_allocator)]
#[cfg(feature="examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() {
    let args: Vec<String> = args().collect();
    let model = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };
    let res = match model {
        None => graph_with_placeholder::run(),
        Some("seq") => seq_nodes::run(),
        Some("vars") => graph_variables::run(),
        Some("lr") => linear_regression::run(),
        Some("lr_py") => linear_regression_from_model::run(),
        Some("conv") => conv_nets::run(),
        Some("conv_mp") => conv_nets_maxpooling::run(),
        Some(_) => graph_with_placeholder::run(),
    };
    // Putting the main code in another function serves two purposes:
    // 1. We can use the `?` operator.
    // 2. We can call exit safely, which does not run any destructors.
    exit(match res {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}