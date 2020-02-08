extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::io;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use crfsuite::{Model, Attribute, CrfError};
use crfsuite::{Trainer, Algorithm, GraphicalModel};

#[derive(Debug, Deserialize, Clone)]
pub struct NER {
    // #[serde(rename = "")]
    // id: String,
    lemma: String,
    #[serde(rename = "next-lemma")]
    next_lemma: String,
    // next-next-lemma: String,
    // next-next-pos: String,
    // next-next-shape: String,
    // next-next-word: String,
    // next-pos: String,
    // next-shape: String,
    // next-word: String,
    // pos: String,
    // prev-iob: String,
    // prev-lemma: String,
    // prev-pos: String,
    // prev-prev-iob: String,
    // prev-prev-lemma: String,
    // prev-prev-pos: String,
    // prev-prev-shape: String,
    // prev-prev-word: String,
    // prev-shape: String,
    // prev-word: String,
    // sentence_idx: String,
    // shape: String,
    word: String,
    tag: String
}

fn get_data() -> Result<Vec<NER>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: NER = result?;
        data.push(r);
    }
    // println!("{:?}", data.len());
    data.shuffle(&mut thread_rng());
    Ok(data)
}

fn split_test_train(data: &[NER], test_size: f32) -> (Vec<NER>, Vec<NER>) {
    let test_size: f32 = data.len() as f32 * test_size;
    let test_size = test_size.round() as usize;

    let (test_data, train_data) = data.split_at(test_size);
    (test_data.to_vec(), train_data.to_vec())
}

fn create_xseq_yseq(data: &[NER]) 
        -> (Vec<Vec<Attribute>>, Vec<String>) {
    let mut xseq = vec![];
    let mut yseq = vec![];
    for item in data {
        let seq = vec![Attribute::new(item.lemma.clone(), 1.0),
            Attribute::new(item.next_lemma.clone(), 0.5)]; // higher weightage for the mainword.
        xseq.push(seq);
        yseq.push(item.tag.clone());
    }
    (xseq, yseq)
}

fn check_accuracy(preds: &[String], actual: &[String]) {
    let mut hits = 0;
    let mut correct_hits = 0;
    for (predicted, actual) in preds.iter().zip(actual) {
        if actual != "O" { // will not consider the other category as it bloats the accuracy.
            if predicted == actual && actual != "O" {
                correct_hits += 1;
            }
            hits += 1;    
        }
    }
    println!("accuracy={} ({}/{} correct)",
        correct_hits as f32 / hits as f32,
        correct_hits,
        hits);
}

fn crfmodel_training(xseq: Vec<Vec<Attribute>>, 
                     yseq: Vec<String>,
                     model_name: &str) -> Result<(), Box<CrfError>> {
    let mut trainer = Trainer::new(true);
    trainer.select(Algorithm::AROW, GraphicalModel::CRF1D)?;
    trainer.append(&xseq, &yseq, 0i32)?;
    trainer.train(model_name, -1i32)?; // using all instances for training.
    Ok(())
}

fn model_prediction(xtest: Vec<Vec<Attribute>>,
                    model_name: &str)
                    -> Result<Vec<String>, Box<CrfError>>{
    let model = Model::from_file(model_name)?;
    let mut tagger = model.tagger()?;
    let preds = tagger.tag(&xtest)?;
    Ok(preds)
}

fn main() {
    let data = get_data().unwrap();
    let (test_data, train_data) = split_test_train(&data, 0.2);
    let (xseq_train, yseq_train) = create_xseq_yseq(&train_data);
    let (xseq_test, yseq_test) = create_xseq_yseq(&test_data);
    crfmodel_training(xseq_train, yseq_train, "rustml.crfsuite").unwrap();
    let preds = model_prediction(xseq_test, "rustml.crfsuite").unwrap();
    check_accuracy(&preds, &yseq_test);
}
