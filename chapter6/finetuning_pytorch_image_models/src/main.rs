#[macro_use]
extern crate failure;
extern crate tch;

use std::ffi::OsStr;
use std::path::Path;

use tch::{Device, Tensor, nn};
use tch::vision::{imagenet, resnet};
use tch::nn::OptimizerConfig;
use tch::vision::imagenet::load_from_dir;

const BATCH_SIZE: i64 = 256;

fn run_model(model_file: &str) -> failure::Fallible<()> {
    Ok(())

}

pub fn main() -> failure::Fallible<()> {

    let args: Vec<_> = std::env::args().collect();
    let (weights, dataset_dir) = match args.as_slice() {
        [_, w, d] => (std::path::Path::new(w), d.to_owned()),
        _ => bail!("usage: main resnet18.ot dataset-path"),
    };
    // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    let dataset = imagenet::load_from_dir(dataset_dir)?;
    println!("{:?}", dataset);

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = resnet::resnet18_no_final_layer(&vs.root());
    vs.load(weights)?;

    // Pre-compute the final activations.
    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());

    let optimizer = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch_idx in 1..1001 {
        let predicted = train_images.apply(&linear);
        let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
        optimizer.backward_step(&loss);

        let test_accuracy = test_images
            .apply(&linear)
            .accuracy_for_logits(&dataset.test_labels);
        println!("{} {:.2}%", epoch_idx, 100. * f64::from(test_accuracy));
    }
    vs.save("model.ot")?;


    // let args: Vec<_> = std::env::args().collect();
    // let model_file = if args.len() < 2 {
    //     bail!("usage: main model.ot")
    // } else {
    //     Some(args[1].as_str())
    // };
    // let model_file = Path::new(model_file.unwrap());
    // let extension = model_file.extension().and_then(OsStr::to_str);
    // match extension {
    //     None => (),
    //     Some("ot") => mnist_linear::run(),
    //     Some("conv") => mnist_conv::run(),
    //     Some(_) => mnist_nn::run(),
    // }
    // // Load the image file and resize it to the usual imagenet dimension of 224x224.
    // let image = imagenet::load_image_and_resize224(image_file)?;

    // // Load the Python saved module.
    // let model = tch::CModule::load(model_file)?;

    // // Apply the forward pass of the model to get the logits.
    // let output = model.forward(&[image.unsqueeze(0)])?.softmax(-1);

    // // Print the top 5 categories for this image.
    // for (probability, class) in imagenet::top(&output, 5).iter() {
    //     println!("{:50} {:5.2}%", class, 100.0 * probability)
    // }

    // let image_dataset = load_from_dir("hymenoptera_data").unwrap();
    // let vs = nn::VarStore::new(Device::cuda_if_available());
    // // let net = Net::new(&vs.root());
    // let net = resnet::resnet18_no_final_layer(&vs.root()); // the model would need to be created.

    // // Pre-compute the final activations.
    // let train_images = tch::no_grad(|| image_dataset.train_images.apply_t(&model, false));
    // let test_images = tch::no_grad(|| image_dataset.test_images.apply_t(&model, false));

    // let vs = nn::VarStore::new(tch::Device::Cpu);
    // let linear = nn::linear(vs.root(), 512, image_dataset.labels, Default::default());
    // let sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    // for epoch_idx in 1..1001 {
    //     let predicted = train_images.apply(&linear);
    //     let loss = predicted.cross_entropy_for_logits(&image_dataset.train_labels);
    //     sgd.backward_step(&loss);

    //     let test_accuracy = test_images
    //         .apply(&linear)
    //         .accuracy_for_logits(&image_dataset.test_labels);
    //     println!("{} {:.2}%", epoch_idx, 100. * f64::from(test_accuracy));
    // }







    // let args: Vec<_> = std::env::args().collect();
    // let (weights, dataset_dir) = match args.as_slice() {
    //     [_, w, d] => (std::path::Path::new(w), d.to_owned()),
    //     _ => bail!("usage: main resnet18.ot dataset-path"),
    // };
    // // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    // let dataset = imagenet::load_from_dir(dataset_dir)?;
    // println!("{:?}", dataset);

    // // // Create the model and load the weights from the file.
    // // let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    // // let net = resnet::resnet18_no_final_layer(&vs.root());
    // // vs.load(weights)?;

    // // Load the Python saved module.
    // let model = tch::CModule::load(weights)?;

    // // // Pre-compute the final activations.
    // // let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    // // let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false)); // this is working
    // let train_images = tch::no_grad(|| model.forward(&[dataset.train_images.unsqueeze(0)]).unwrap().softmax(-1));
    // let test_images = tch::no_grad(|| model.forward(&[dataset.test_images.unsqueeze(0)]).unwrap().softmax(-1));

    // let vs = nn::VarStore::new(tch::Device::Cpu);
    // let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    // let sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    // for epoch_idx in 1..1001 {
    //     let predicted = train_images.apply(&linear);
    //     let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
    //     sgd.backward_step(&loss);

    //     let test_accuracy = test_images
    //         .apply(&linear)
    //         .accuracy_for_logits(&dataset.test_labels);
    //     println!("{} {:.2}%", epoch_idx, 100. * f64::from(test_accuracy));
    // }





    // let args: Vec<_> = std::env::args().collect();
    // let (model_file, image_file) = match args.as_slice() {
    //     [_, m, i] => (m.to_owned(), i.to_owned()),
    //     _ => bail!("usage: main model.pt image.jpg"),
    // };
    // // Load the image file and resize it to the usual imagenet dimension of 224x224.
    // let image = imagenet::load_image_and_resize224(image_file)?;

    // // Load the Python saved module.
    // let model = tch::CModule::load(model_file)?;

    // // Apply the forward pass of the model to get the logits.
    // let output = image.unsqueeze(0).apply(&model).softmax(-1);

    // // Print the top 5 categories for this image.
    // for (probability, class) in imagenet::top(&output, 5).iter() {
    //     println!("{:50} {:5.2}%", class, 100.0 * probability)
    // }

    // let dataset = load_from_dir("hymenoptera_data").unwrap();

    // // // Pre-compute the final activations.
    // // let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    // // let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false)); // this is working
    // let train_images = tch::no_grad(|| dataset.train_images.unsqueeze(0).apply(&model).softmax(-1));
    // let test_images = tch::no_grad(|| dataset.test_images.unsqueeze(0).apply(&model).softmax(-1));
    // println!("{:?}", train_images.size());

    // let vs = nn::VarStore::new(tch::Device::Cpu);
    // let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    // let sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    // for epoch_idx in 1..1001 {
    //     let predicted = train_images.apply(&linear);
    //     let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
    //     sgd.backward_step(&loss);

    //     let test_accuracy = test_images
    //         .apply(&linear)
    //         .accuracy_for_logits(&dataset.test_labels);
    //     println!("{} {:.2}%", epoch_idx, 100. * f64::from(test_accuracy));
    // }

    Ok(())
}