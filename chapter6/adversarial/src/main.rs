// CNN model. This should rearch 99.1% accuracy.

use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view(&[-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view(&[-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout_(0.5, train)
            .apply(&self.fc2)
    }
}

// FGSM attack code
fn fgsm_attack(image: &Tensor, epsilon: f64, data_grad: &Tensor) -> Tensor {
    // Collect the element-wise sign of the data gradient
    let sign_data_grad = data_grad.sign();
    // Create the perturbed image by adjusting each pixel of the input image
    // let perturbed_image = image + epsilon*sign_data_grad;
    let change = sign_data_grad * epsilon;
    let mut perturbed_image = image + change;
    // # Adding clipping to maintain [0,1] range
    let perturbed_image = perturbed_image.clamp_(0., 1.);
    // # Return the perturbed image
    perturbed_image
}

pub fn main() -> failure::Fallible<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());
    let opt = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..100 {
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let bimages = bimages.set_requires_grad(true);
            println!("{:?}", bimages.requires_grad());
            
            let data_grad = bimages.grad();
            // println!("{:?}", data_grad.sign());

            // # Call FGSM Attack
            let epsilon = 0.5;
            let perturbed_data = fgsm_attack(&bimages, epsilon, &data_grad);

            let loss = net
                .forward_t(&perturbed_data, true)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
    Ok(())
}