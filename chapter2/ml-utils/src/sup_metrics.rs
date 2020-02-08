use std::cmp::Ordering;

// for regression
pub fn r_squared_score(y_test: &[f64], y_preds: &[f64]) -> f64 {
    let model_variance: f64 = y_test.iter().zip(y_preds.iter()).fold(
        0., |v, (y_i, y_i_hat)| {
            v + (y_i - y_i_hat).powi(2)
        }
    );

    // get the mean for the actual values to be used later
    let y_test_mean = y_test.iter().sum::<f64>() as f64
        / y_test.len() as f64;

    // finding the variance
    let variance =  y_test.iter().fold(
        0., |v, &x| {v + (x - y_test_mean).powi(2)}
    );
    let r2_calculated: f64 = 1.0 - (model_variance / variance);
    r2_calculated
}

// for classification
pub fn accuracy(y_test: &[u32], y_preds: &[u32]) -> f32 {
    let mut correct_hits = 0;
    for (predicted, actual) in y_preds.iter().zip(y_test.iter()) {
        if predicted == actual {
            correct_hits += 1;
        }
    }
    let acc: f32 = correct_hits as f32 / y_test.len() as f32;
    acc
}

pub fn logloss_score(y_test: &[f32], y_preds: &[f32], eps: f32) -> f32 {
    // complete this http://wiki.fast.ai/index.php/Log_Loss#Log_Loss_vs_Cross-Entropy
    let y_preds = y_preds.iter().map(|&p| {
        match p.partial_cmp(&(1.0 - eps)) {
            Some(Ordering::Less) => p,
            _ => 1.0 - eps, // if equal or greater.
        }
    });
    let y_preds = y_preds.map(|p| {
        match p.partial_cmp(&eps) {
            Some(Ordering::Less) => eps,
            _ => p,
        }
    });

    // Now compute the logloss
    let logloss_vals = y_preds.zip(y_test.iter()).map(|(predicted, &actual)| {
        if actual as f32 == 1.0 {
            (-1.0) * predicted.ln()
        } else if actual as f32 == 0.0 {
            (-1.0) * (1.0 - predicted).ln()
        } else {
            panic!("Invalid labels: target data is not either 0.0 or 1.0");
        }
    });
    logloss_vals.sum()
}