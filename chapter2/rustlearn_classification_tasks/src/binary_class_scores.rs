use std::error::Error;

use rustlearn::prelude::*;
use rustlearn::metrics::{accuracy_score, roc_auc_score};

use ml_utils;
use ml_utils::sup_metrics::{accuracy, logloss_score};

pub fn run() -> Result<(), Box<dyn Error>> {
    let preds = vec![1., 0.0001, 0.908047338626, 0.0199900075962, 0.904058545833, 0.321508119045, 0.657086320195];
    let actuals = vec![1., 0., 0., 1., 1., 0., 0.];
    println!("logloss score: {:?}", logloss_score(&actuals, &preds, 1e-15));
    println!("roc auc scores: {:?}", roc_auc_score(&Array::from(actuals), &Array::from(preds))?);

    Ok(())
}