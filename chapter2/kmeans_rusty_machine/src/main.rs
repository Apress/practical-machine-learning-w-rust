extern crate rusty_machine;
extern crate rand;

use rusty_machine::linalg::{Matrix, BaseMatrix};
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::UnSupModel;

use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use csv;
use ml_utils;
use ml_utils::datasets::Flower;

// fn generate_data(centroids: &Matrix<f64>,
//                  points_per_centroid: usize,
//                  noise: f64)
//                  -> Matrix<f64> {
//     assert!(centroids.cols() > 0, "Centroids cannot be empty.");
//     assert!(centroids.rows() > 0, "Centroids cannot be empty.");
//     assert!(noise >= 0f64, "Noise must be non-negative.");
//     let mut raw_cluster_data = Vec::with_capacity(centroids.rows() * points_per_centroid *
//                                                   centroids.cols());

//     let mut rng = thread_rng();
//     let normal_rv = Normal::new(0f64, noise);

//     for _ in 0..points_per_centroid {
//         // Generate points from each centroid
//         for centroid in centroids.row_iter() {
//             // Generate a point randomly around the centroid
//             let mut point = Vec::with_capacity(centroids.cols());
//             for feature in centroid.iter() {
//                 point.push(feature + normal_rv.ind_sample(&mut rng));
//             }

//             // Push point to raw_cluster_data
//             raw_cluster_data.extend(point);
//         }
//     }

//     Matrix::new(centroids.rows() * points_per_centroid,
//                 centroids.cols(),
//                 raw_cluster_data)
// }

fn main() {
    let data = "sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n";
    let mut rdr = csv::Reader::from_reader(data.as_bytes());
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: Flower = result.unwrap();
        data.push(r); // data contains all the records
    }

    let flower_x_train: Vec<f32> = data.iter().map(|r| r.into_features()).collect();
    let flower_x_train = Matrix::new(flower_x_train.len, 4, flower_x_train);


    println!("K-Means clustering example:");

    const SAMPLES_PER_CENTROID: usize = 2000;

    println!("Generating {0} samples from each centroids:",
             SAMPLES_PER_CENTROID);
    // Choose two cluster centers, at (-0.5, -0.5) and (0, 0.5).
    let centroids = Matrix::new(2, 4, vec![-0.5, -0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0]);
    println!("{}", centroids);

    // // Generate some data randomly around the centroids
    // let samples = generate_data(&centroids, SAMPLES_PER_CENTROID, 0.4);

    // Create a new model with 2 clusters
    let mut model = KMeansClassifier::new(2);

    // Train the model
    println!("Training the model...");
    // Our train function returns a Result<(), E>
    model.train(&flower_x_train).unwrap();

    let centroids = model.centroids().as_ref().unwrap();
    println!("Model Centroids:\n{:.3}", centroids);

    // Predict the classes and partition into
    println!("Classifying the samples...");
    let classes = model.predict(&flower_x_train).unwrap();
    let (first, second): (Vec<usize>, Vec<usize>) = classes.data().iter().partition(|&x| *x == 0);

    println!("Samples closest to first centroid: {}", first.len());
    println!("Samples closest to second centroid: {}", second.len());
}
