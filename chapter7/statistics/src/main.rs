#![allow(non_snake_case)]

use std::path::Path;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use std::f64;

use ndarray;
use ndarray::{Array, Array2, Array1, Axis, stack};

// https://nbviewer.jupyter.org/urls/umich.box.com/shared/static/7kh8amlez7bx3qlqa6aa.ipynb?create=1

fn process_gene_expresssion_data_headers(thisline: &String, SIF: &HashMap<String, String>) -> (Vec<String>, Vec<String>, Vec<usize>) {
    let SID: Vec<String> = thisline.split("\t").map(|s| s.to_owned()).collect();
    let indices: Vec<usize> = SID.iter()
                .enumerate()
                .filter(|&(_, x)| x.starts_with("GSM") )
                .map(|(i, _)| i).collect();
    let SID: Vec<String> = indices.iter().map(|&i| SID[i].clone()).collect();
    let STP: Vec<String> = SID.iter().map(
        |k| SIF.get(&k.to_string()).unwrap())
        .cloned().collect();
    (SID, STP, indices)
}

/// Read the gene expression data as a list of lists, also get the gene identifiers
fn process_gene_expresssion_data(thisline: &String, indices: &Vec<usize>) -> (Vec<f64>, String) {
    let gene_expression_measure_values: Vec<String> = thisline.split("\t")
        .map(|s| s.to_owned()).collect();
    let gene_expression_measures = indices.iter().map(
        |&i| gene_expression_measure_values[i].clone());
    let gene_expression_measures: Vec<f64> = gene_expression_measures.map(
        |x| x.parse().unwrap()).collect();
    let gene_identifiers = vec![gene_expression_measure_values[0].clone(),
        gene_expression_measure_values[1].clone()].join(";");
    (gene_expression_measures, gene_identifiers)
}

fn filter_specific_samples(STP: &Vec<String>, group_type: &str) -> Vec<usize> {
    STP.iter().enumerate()
        .filter(|&(_, x)| x == group_type )
        .map(|(i, _)| i).collect()
}

fn different_samples(STP: &Vec<String>) -> (Vec<usize>, Vec<usize>) {
    let UC = filter_specific_samples(&STP, "ulcerative colitis");
    let CD = filter_specific_samples(&STP, "Crohn's disease");
    (UC, CD)
}

fn convert_to_log_scale(X: &Array2<f64>) -> Array2<f64> {
    let two = 2.0f64;
    let two_log = two.ln();
    X.mapv(|x| x.ln()/two_log)
}

fn filter_out_relevant_columns(samples: &Vec<usize>, gene_expression_measures_matrix: &Array2<f64>) -> Array2<f64> {
    let shape1 = samples.len();
    let shape0 = gene_expression_measures_matrix.shape()[0];
    let mut cols = Vec::new();
    for &msamples_columns in samples {
        let col = gene_expression_measures_matrix.column(msamples_columns);
        cols.push(col);
    }
    let Msamples = stack(Axis(0), &cols[..]).unwrap();
    let Msamples = Array::from_iter(Msamples.iter());
    let Msamples = Msamples.into_shape((shape0, shape1)).unwrap();
    Msamples.mapv(|&x| x)
}

fn mean_of_samples(samples: &Vec<usize>, gene_expression_measures_matrix: &Array2<f64>) -> Array1<f64> {
    let Msamples = filter_out_relevant_columns(samples, gene_expression_measures_matrix);
    Msamples.mean_axis(Axis(1))
}

fn variance_of_samples(
        samples: &Vec<usize>, gene_expression_measures_matrix: &Array2<f64>)
        -> Array1<f64> {
    let Msamples = filter_out_relevant_columns(samples, gene_expression_measures_matrix);
    Msamples.var_axis(Axis(1), 1.)
}

fn generate_zscores(UC: &Vec<usize>,
                    CD: &Vec<usize>,
                    gene_expression_measures_matrix: &Array2<f64>)
                    -> Array1<f64> {
    let MUC = mean_of_samples(&UC, &gene_expression_measures_matrix);
    let MCD = mean_of_samples(&CD, &gene_expression_measures_matrix);
    let VUC = variance_of_samples(&UC, &gene_expression_measures_matrix);
    let VCD = variance_of_samples(&CD, &gene_expression_measures_matrix);
    let nUC = UC.len();
    let nCD = CD.len();
    let z_scores_num = MUC - MCD;
    let z_scores_den = (VUC/nUC as f64 + VCD/nCD as f64).mapv(f64::sqrt);
    let z_scores = z_scores_num / z_scores_den;
    z_scores
}


fn process_file(filename: &Path) -> io::Result<HashMap<String, String>> {
    let mut SIF = HashMap::new();
    let file = File::open(filename).unwrap();
    let mut subset_description = String::new();
    let mut within_dataset_table = false;
    let mut within_headers = true;
    let mut gene_expression_headers = true;
    let mut indices: Vec<usize> = Vec::new();
    let mut gene_expression_measures_vec = Vec::new();
    let mut gene_identifiers_vec = Vec::new();
    let mut SID = Vec::new();
    let mut STP = Vec::new();
    'linereading: for line in BufReader::new(file).lines() {
        let thisline = line?;
        let line_split: Vec<String> = thisline.split("=").map(|s| s.to_owned()).collect();
        if thisline.starts_with("!dataset_table_begin") {
            within_dataset_table = true;
            within_headers = false;
            continue 'linereading;
        }
        if within_dataset_table && gene_expression_headers {
            // println!("wihting within_dataset_table && gene_expression_headers", );
            let sid_stp_indices = process_gene_expresssion_data_headers(&thisline, &SIF);
            indices = sid_stp_indices.2.clone();
            SID = sid_stp_indices.0.clone();
            STP = sid_stp_indices.1.clone();
            gene_expression_headers = false;
            continue 'linereading;
        };
        // println!("indices: {:?}", indices);
        if within_dataset_table && !gene_expression_headers {
            if thisline.starts_with("!dataset_table_end") {
                break 'linereading;
            }
            let (gene_expression_measures, gene_identifiers) = process_gene_expresssion_data(&thisline, &indices);
            gene_expression_measures_vec.extend(gene_expression_measures);
            gene_identifiers_vec.push(gene_identifiers);
        }
        if within_headers {
            if thisline.starts_with("!subset_description") {
                subset_description = line_split[1].trim().to_owned();
            };
            let subset_ids = if thisline.starts_with("!subset_sample_id") {
                let subset_ids = line_split[1].split(",");
                let subset_ids = subset_ids.map(|s| s.trim().to_owned());
                subset_ids.collect()
            } else {
                Vec::new()
            };
            for k in subset_ids {
                SIF.insert(k, subset_description.to_owned());
                // println!("SIF: {:?}", SIF);
            }
        }
    }
    let gene_expression_measures_matrix = Array::from_shape_vec((22283, 127), gene_expression_measures_vec).unwrap();
    let gene_expression_measures_matrix = convert_to_log_scale(&gene_expression_measures_matrix);
    let (UC, CD) = different_samples(&STP);
    let z_scores = generate_zscores(&UC, &CD, &gene_expression_measures_matrix);
    let z_scores_mean = z_scores.sum() / z_scores.len() as f64;
    let z_scores_std = z_scores.std_axis(Axis(0), 1.);
    println!("z scores mean {:?}", z_scores_mean);
    println!("z scores mean {:?}", z_scores_std);

    Ok(SIF)
}

fn main() {
    let filename = Path::new("GDS1615_full.soft");
    process_file(&filename).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_mean_of_samples() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let res = arr1(&[2.5, 3.5]);
        let samples = vec![0, 1];
        assert_eq!(mean_of_samples(&samples, &a), res);
    }
}