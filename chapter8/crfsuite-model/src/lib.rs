extern crate serde;
#[macro_use]
extern crate serde_derive;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::fs;
use std::path::PathBuf;

use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use crfsuite::{Model, Attribute, CrfError};
use crfsuite::{Trainer, Algorithm, GraphicalModel};

#[pyclass(module = "crfsuite_model")]
pub struct CRFSuiteModel {
    model_name: String,
}

#[pymethods]
impl CRFSuiteModel {
    #[new]
    fn new(obj: &PyRawObject, path: String) {
        obj.init(CRFSuiteModel {
            model_name: path,
        });
    }

    fn fit(&self, py: Python<'_>, path: String) -> PyResult<String> {
        let data_file = PathBuf::from(&path[..]);
        let data = get_data(&data_file).unwrap();
        let (test_data, train_data) = split_test_train(&data, 0.2);
        let (xseq_train, yseq_train) = create_xseq_yseq(&train_data);
        let (xseq_test, yseq_test) = create_xseq_yseq(&test_data);
        crfmodel_training(xseq_train, yseq_train, self.model_name.as_ref()).unwrap();
        let preds = model_prediction(xseq_test, self.model_name.as_ref()).unwrap();
        check_accuracy(&preds, &yseq_test);
        Ok("model fit done".to_string())
    }

    fn predict(&self, predict_filename: String) -> PyResult<Vec<String>> {
        let predict_data_file = PathBuf::from(predict_filename);
        let data = get_data_no_y(&predict_data_file).unwrap();
        let xseq_test = create_xseq_for_predict(&data[..]);
        let preds = model_prediction(xseq_test, self.model_name.as_ref()).unwrap();
        Ok(preds)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct NER {
    lemma: String,
    #[serde(rename = "next-lemma")]
    next_lemma: String,
    word: String,
    tag: String
}

#[derive(Debug, Deserialize, Clone)]
pub struct NER_Only_X {
    lemma: String,
    #[serde(rename = "next-lemma")]
    next_lemma: String,
    word: String,
}

fn get_data_no_y(path: &PathBuf) -> Result<Vec<NER_Only_X>, Box<dyn Error>> {
    let csvfile = fs::File::open(path)?;
    let mut rdr = csv::Reader::from_reader(csvfile);
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: NER_Only_X = result?;
        data.push(r);
    }
    Ok(data)
}

fn get_data(path: &PathBuf) -> Result<Vec<NER>, Box<dyn Error>> {
    let csvfile = fs::File::open(path)?;
    let mut rdr = csv::Reader::from_reader(csvfile);
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: NER = result?;
        data.push(r);
    }
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

fn create_xseq_for_predict(data: &[NER_Only_X])
        -> Vec<Vec<Attribute>> {
    let mut xseq = vec![];
    for item in data {
        let seq = vec![Attribute::new(item.lemma.clone(), 1.0),
            Attribute::new(item.next_lemma.clone(), 0.5)]; // higher weightage for the mainword.
        xseq.push(seq);
    }
    xseq
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

#[pymodule]
fn crfsuite_model(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CRFSuiteModel>()?;

    Ok(())
}