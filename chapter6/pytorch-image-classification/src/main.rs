/// Module for showing pytorch image classification
/// 
/// To run this can use `cargo run`
extern crate tch;

use std::process::exit;
use std::env::args;
use std::io;
use std::fs::{self, DirEntry, copy, create_dir_all};
use std::path::Path;

use tch::{Device, Tensor, nn};
use tch::nn::{ModuleT, OptimizerConfig, conv2d, linear};
use tch::vision::imagenet::{load_from_dir, load_image_and_resize224, top};
use failure;

// for the CNN
const BATCH_SIZE: i64 = 16;
// const LABELS: i64 = 10;
const LABELS: i64 = 102;
// const LABELS: i64 = 6;
const EPOCHS: i64 = 2;

const W: i64 = 224;
const H: i64 = 224;
const C: i64 = 3;

// for the simple nn
const IMAGE_DIM: i64 = W * W * C;
const HIDDEN_NODES: i64 = 128;

const DATASET_FOLDER: &str = "dataset";

/// Visit directory, identify the train and test files
/// and run the train_fn on the training files
/// and the test function on the test files.
/// 
/// # Arguments
/// * `dir` - The directory that this function should run on.
/// * `train_fn` - Training function for the training files.
/// * `test_fn` - Testing function for the test files.
fn visit_dir(dir: &Path,
             train_fn: &dyn Fn(&DirEntry),
             test_fn: &dyn Fn(&DirEntry))
             -> io::Result<()> {
    if dir.is_dir() {
        let mut this_label = String::from("");
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dir(&path, train_fn, test_fn)?;
            } else {
                let full_path: Vec<String> = path.to_str().unwrap()
                    .split("/").into_iter()
                    .map(|x| x[..].to_string()).collect();
                if this_label == full_path[1] {
                    train_fn(&entry); // move the training file
                } else {
                    test_fn(&entry); // move the testing file
                }
                // the second entry is the label
                this_label = full_path[1].clone();
            }
        }
    }
    Ok(())
}

/// A simple print statement to be utilised for testing reasons mostly.
fn print_directory(dir: &Path) {
    println!("{:?}", dir);
}

/// move files from source to destination
/// 
/// # Arguments
/// * `from_path`: the directory that needs to be copied
/// * `to_path`: the target directory where the file contents should be placed.
/// 
/// # Returns
/// Result if the copy is successful else raises error
fn move_file(from_path: &DirEntry, to_path: &Path) -> io::Result<()> {
    let root_folder = Path::new(DATASET_FOLDER);
    let second_order = root_folder.join(to_path);
    let full_path: Vec<String> = from_path.path().to_str().unwrap()
        .split("/").into_iter().map(|x| x[..].to_string()).collect();
    let label = full_path[1].clone();
    let third_order = second_order.join(label);
    create_dir_all(&third_order)?;
    let filename = from_path.file_name();
    let to_filename = third_order.join(&filename);
    // println!("{:?}", to_filename);
    copy(from_path.path(), to_filename)?;
    Ok(())
}

// image dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

/// A convolutional net is represented here.
#[derive(Debug)]
struct CnnNet {
    /// the first convolutional layer
    conv1: nn::Conv2D,
    /// the second convolutional layer
    conv2: nn::Conv2D,
    /// a linear layer to flatten the previous convolution
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl CnnNet {
    fn new(vs: &nn::Path) -> CnnNet {
        let conv1 = conv2d(vs, C, 32, 5, Default::default());
        let conv2 = conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = linear(vs, 179776, 1024, Default::default());
        // let fc1 = linear(vs, 1024, 1024, Default::default());
        let fc2 = linear(vs, 1024, LABELS, Default::default());
        CnnNet {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for CnnNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view(&[-1, C, H, W])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view(&[BATCH_SIZE, -1])
            .apply(&self.fc1)
            .relu()
            .dropout_(0.5, train)
            .apply(&self.fc2)
    }
}

// impl nn::ModuleT for CnnNet {
//    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
//        let xs_prime = xs.view(&[-1, C, H, W]);
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.apply(&self.conv1);
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.max_pool2d_default(2);
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.apply(&self.conv2);
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.max_pool2d_default(2);
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.view(&[BATCH_SIZE, -1]);
//     //    let xs_prime = xs_prime.view(&[-1, 1024]);
//        println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.apply(&self.fc1);
//        println!("{:?}", xs_prime.size());
//        println!("%%%%%%%%%%%", );
//        let mut xs_prime = xs_prime.relu();
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.dropout_(0.5, train);
//     //    println!("{:?}", xs_prime.size());
//        let xs_prime = xs_prime.apply(&self.fc2);
//     //    println!("{:?}", xs_prime.size());
//        xs_prime
//    }
// }

fn learning_rate(epoch: i64) -> f64 {
    if epoch < 10 {
        0.1
    } else if epoch < 20 {
        0.01
    } else {
        1e-4
    }
}

fn main() -> failure::Fallible<()> {
    let args: Vec<String> = args().collect();
    let create_directories = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };
    match create_directories {
        None => (),
        Some("yes") => {
            let obj_categories = Path::new("101_ObjectCategories");
            let move_to_train = |x: &DirEntry| {
               let to_folder = Path::new("train");
               move_file(&x, &to_folder).unwrap();
            };
            let move_to_test =
                |x: &DirEntry| {
                    let to_folder = Path::new("val");
                    move_file(&x, &to_folder).unwrap();
                };
            visit_dir(
                &obj_categories, &move_to_train, &move_to_test).unwrap();
        },
        Some(_) => {
            println!("Usage: cargo run yes", );
            exit(1)
        },
    }
    println!("files kept in the imagenet format in {}", DATASET_FOLDER);
    println!("moving on with training.");
    let image_dataset = load_from_dir(DATASET_FOLDER).unwrap();
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let optimizer = nn::Adam::default().build(&vs, 1e-4)?;
    let net = CnnNet::new(&vs.root());
    for epoch in 1..EPOCHS+1 {
        for (bimages, blabels) in image_dataset.train_iter(BATCH_SIZE).shuffle().to_device(vs.device()) {
            // println!("images and labels size {:?}, {:?}", bimages.size(), blabels.size());
            // let outputs = net
            //     .forward_t(&bimages, true);
            // println!("outputs done {:?}", outputs.size());
            // let loss = outputs
            //     .cross_entropy_for_logits(&blabels);
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            optimizer.backward_step(&loss);
        }
        // println!("training done");
        // println!("test size {:?}", image_dataset.test_images.size());
        // println!("test size {:?}", image_dataset.test_labels.size());
        let test_accuracy =
            net.batch_accuracy_for_logits(&image_dataset.test_images,
                                          &image_dataset.test_labels,
                                          vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%",
            epoch, 100. * test_accuracy,);
    }
    // model saving
    vs.save("model.ot")?;
    println!("Training done!");



    // model loading and prediction
    let weights = Path::new("model.ot");
    let image = "image.jpg"; // save an image in the path.

    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = load_image_and_resize224(image)?;

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net: Box<dyn ModuleT> = Box::new(CnnNet::new(&vs.root()));
    vs.load(weights)?;

    // Apply the forward pass of the model to get the logits.
    let output = net
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1); // Convert to probability.

    // Print the top 5 categories for this image.
    for (probability, class) in top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}
