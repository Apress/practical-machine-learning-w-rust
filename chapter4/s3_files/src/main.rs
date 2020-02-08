use std::env;
use std::fs::File;
use std::io::Read;
use std::str;
use std::vec::Vec;
use std::error::Error;

use env_logger;
use futures::{Future, Stream};
// use rusoto;
// use rusoto::s3;
use rusoto_core;
use rusoto_core::credential::{AwsCredentials, DefaultCredentialsProvider};
use rusoto_core::{Region, ProvideAwsCredentials, RusotoError};
use rusoto_s3::{
    CreateBucketRequest, DeleteBucketRequest,
    DeleteObjectRequest, GetObjectRequest, ListObjectsV2Request,
    PutObjectRequest, S3Client, S3,
};
use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;
use rustlearn::prelude::*;
use rustlearn::svm::libsvm::svc::{Hyperparameters as libsvm_svc, KernelType};


use ml_utils;
use ml_utils::datasets::Flower;

fn list_s3_buckets(client: &S3Client) {
    let result = client.list_buckets().sync().expect("Couldn't list buckets");
    println!("\nbuckets available: {:#?}", result);
}

fn create_s3_bucket(client: &S3Client, bucket: &str) {
    let create_bucket_req = CreateBucketRequest {
        bucket: bucket.to_owned(),
        ..Default::default()
    };

    client
        .create_bucket(create_bucket_req)
        .sync()
        .expect("Couldn't create bucket");
}

fn list_items_in_bucket(client: &S3Client, bucket: &str) {
    let list_obj_req = ListObjectsV2Request {
        bucket: bucket.to_owned(),
        start_after: Some("foo".to_owned()),
        ..Default::default()
    };
    let result = client
        .list_objects_v2(list_obj_req)
        .sync()
        .expect("Couldn't list items in bucket (v2)");
    println!("Items in bucket: {:#?}", result);
}

fn push_file_to_s3(
    client: &S3Client,
    bucket: &str,
    dest_filename: &str,
    local_filename: &str,
) {
    let mut f = File::open(local_filename).unwrap();
    let mut contents: Vec<u8> = Vec::new();
    match f.read_to_end(&mut contents) {
        Err(why) => panic!("Error opening file to send to S3: {}", why),
        Ok(_) => {
            let req = PutObjectRequest {
                bucket: bucket.to_owned(),
                key: dest_filename.to_owned(),
                body: Some(contents.into()),
                ..Default::default()
            };
            client.put_object(req).sync().expect("Couldn't PUT object");
        }
    }
}

fn delete_s3_file(client: &S3Client, bucket: &str, filename: &str) {
    let del_req = DeleteObjectRequest {
        bucket: bucket.to_owned(),
        key: filename.to_owned(),
        ..Default::default()
    };

    client
        .delete_object(del_req)
        .sync()
        .expect("Couldn't delete object");
}

fn delete_s3_bucket(client: &S3Client, bucket: &str) {
    let delete_bucket_req = DeleteBucketRequest {
        bucket: bucket.to_owned(),
        ..Default::default()
    };

    let result = client.delete_bucket(delete_bucket_req).sync();
    match result {
        Err(e) => match e {
            RusotoError::Unknown(ref e) => panic!(
                "Couldn't delete bucket because: {}",
                str::from_utf8(&e.body).unwrap()
            ),
            _ => panic!("Error from S3 different than expected"),
        },
        Ok(_) => (),
    }
}

fn pull_object_from_s3(client: &S3Client, bucket: &str, filename: &str) -> Result<String, Box<Error>> {
    let get_req = GetObjectRequest {
        bucket: bucket.to_owned(),
        key: filename.to_owned(),
        ..Default::default()
    };

    let result = client
        .get_object(get_req)
        .sync()
        .expect("Couldn't GET object");
    println!("get object result: {:#?}", result);

    let stream = result.body.unwrap();
    let body = stream.concat2().wait().unwrap();

    Ok(str::from_utf8(&body)?.to_owned())
}

fn main() -> Result<(), Box<Error>> {
    let _ = env_logger::try_init();

    let region = if let Ok(endpoint) = env::var("S3_ENDPOINT") {
        let region = Region::Custom {
            // name: "us-east-1".to_owned(),
            name: "ap-south-1".to_owned(),
            endpoint: endpoint.to_owned(),
        };
        println!(
            "picked up non-standard endpoint {:?} from S3_ENDPOINT env. variable",
            region
        );
        region
    } else {
        // Region::UsEast1
        Region::ApSouth1
    };
    let credentials = DefaultCredentialsProvider::new()
        .unwrap()
        .credentials()
        .wait()
        .unwrap();
    let client = S3Client::new(region.clone());
    let s3_bucket = format!("rust-ml-bucket");
    let filename = format!("iris.csv");

    // list_s3_buckets(&client);
    // create_s3_bucket(&client, &s3_bucket);
    // list_items_in_bucket(&client, &s3_bucket);
    push_file_to_s3(&client, &s3_bucket, &filename, "data/iris.csv");
    let data = pull_object_from_s3(&client, &s3_bucket, &filename)?;

    // go ahead with the csv module
    let mut rdr = csv::Reader::from_reader(data.as_bytes());
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: Flower = result?;
        data.push(r);
    }

    // shuffle the data.
    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets.
    let test_size: f32 = 0.2;
    let test_size: f32 = data.len() as f32 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the labels.
    let flower_x_train: Vec<f32> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_train: Vec<f32> = train_data.iter().map(|r| r.into_labels()).collect();

    let flower_x_test: Vec<f32> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let flower_y_test: Vec<f32> = test_data.iter().map(|r| r.into_labels()).collect();

    let mut flower_x_train = Array::from(flower_x_train);
    flower_x_train.reshape(train_size, 4);

    let flower_y_train = Array::from(flower_y_train);

    let mut flower_x_test = Array::from(flower_x_test);
    flower_x_test.reshape(test_size, 4);

    // Working with svms
    let mut svm_rbf_model = libsvm_svc::new(4, KernelType::RBF, 3)
        .C(0.3)
        .build();
    svm_rbf_model.fit(&flower_x_train, &flower_y_train).unwrap();
    println!("SVM Model training done");

    delete_s3_file(&client, &s3_bucket, &filename);
    println!("Deleted: s3://{bucket}/{fn}", bucket=s3_bucket, fn=filename);
    // delete_s3_bucket(&client, &s3_bucket);
    // println!("Deleted: s3://{bucket}", bucket=s3_bucket);


    Ok(())
}
