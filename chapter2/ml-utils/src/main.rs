use std::vec::Vec;
use std::collections::HashSet;
use std::iter::FromIterator;

use rand;
use itertools;
use itertools::iproduct;
use itertools::Itertools;
use ndarray;
use ndarray::{arr2, Array, ArrayBase, OwnedRepr, Dim, Axis};
use ndarray::prelude::*;

// reference: https://github.com/Hoosier-Clusters/clusim/blob/master/clusim/sim.py

fn matching_elems_count(s1: &HashSet<u8>, s2: &HashSet<u8>) -> u8 {
    let common: Vec<_> = s1.intersection(s2).collect();
    common.len() as u8
}

fn contingency_table(clusters1: &Vec<HashSet<u8>>, clusters2: &Vec<HashSet<u8>>) -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>> {
    let length = clusters1.len();
    assert!(length == clusters2.len());
    let product = iproduct!(clusters1, clusters2);
    let cont_table_vec: Vec<u8> = product.map(
        |(c1, c2)| matching_elems_count(c1, c2)
    ).collect();
    // println!("{:?}", cont_table_vec);
    let cont_table_mat = Array::from_shape_vec((3, 3), cont_table_vec).unwrap();
    cont_table_mat
    // let v_chunked: Vec<Vec<f64>> = cont_table_vec.chunks(length).map(|x| x.to_vec()).collect();
    // v_chunked
}

fn cluster_size_sequence_sqsum(clusters: &Vec<HashSet<u8>>) -> u8 {
    let cluster1_size_seq: Vec<u8> = clusters.iter().map(
        |v| v.len() as u8).collect();
    let squares = cluster1_size_seq.iter().map(
        |num| num.pow(2)
    );
    squares.sum()

}

fn elements_in_vectr(vectr: &Vec<HashSet<u8>>) -> u8 {
    let flatten_array: Vec<u8> = vectr
        .iter()
        .flat_map(|array| array.iter())
        .cloned()
        .collect();
    flatten_array.len() as u8

}

fn count_pairwise_cooccurence(clusters1: &Vec<HashSet<u8>>,
        clusters2: &Vec<HashSet<u8>>) -> (f64, f64, f64, f64) {
    let cont_tbl = contingency_table(&clusters1, &clusters2);
    // println!("{:?}", cont_tbl);

    let square_matrix = cont_tbl.mapv(|a| a.pow(2));
    // println!("{:?}", square_matrix);
    let sum_of_squares = square_matrix.sum();
    // println!("{:?}", sum_of_squares);
    let c1_sum_sq_sizes = cluster_size_sequence_sqsum(clusters1);
    let c2_sum_sq_sizes = cluster_size_sequence_sqsum(clusters2);
    // println!("{:?}", c1_sum_sq_sizes);

    let c1_elements_count = elements_in_vectr(clusters1);
    let n11 = 0.5 * (sum_of_squares - c1_elements_count) as f64;
    // println!("{:?}", n11);
    let n10 = 0.5 * (c1_sum_sq_sizes - sum_of_squares) as f64;
    let n01 = 0.5 * (c2_sum_sq_sizes - sum_of_squares) as f64;
    let n00 = 0.5 * c1_elements_count as f64 * (c1_elements_count - 1) as f64 - n11 - n10 - n01;
    (n11, n10, n01, n00)
}

fn hashset(data: &[u8]) -> HashSet<u8> {
    HashSet::from_iter(data.iter().cloned())
}

fn jaccard_index(clusters1: &Vec<HashSet<u8>>, clusters2: &Vec<HashSet<u8>>) -> f64 {
    let (n11, n10, n01, n00) = count_pairwise_cooccurence(clusters1, clusters2);
    println!("{:?}", (n11, n10, n01, n00));
    let denominator = n11 + n10 + n01;
    if denominator > 0.0 {
        return n11 / denominator;
    } else {
        0.0
    }
}

fn rand_index(clusters1: &Vec<HashSet<u8>>, clusters2: &Vec<HashSet<u8>>) -> f64 {
    let (n11, n10, n01, n00) = count_pairwise_cooccurence(clusters1, clusters2);
    (n11 + n00) / (n11 + n10 + n01 + n00)
}

fn main() {
    let clusters1 = vec![vec![0u8,8,3, 7], vec![1u8,5], vec![2u8, 4, 6]];
    let clusters1: Vec<HashSet<u8>> = clusters1.iter().map(
        |v| hashset(&v)).collect();
    let clusters2 = vec![vec![0u8,4, 7], vec![1u8,2,3, 6], vec![8u8,5]];
    let clusters2: Vec<HashSet<u8>> = clusters2.iter().map(
        |v| hashset(&v)).collect();

    let ji = jaccard_index(&clusters1, &clusters2);
    println!("jaccard index: {:?}", ji);

    let ri = rand_index(&clusters1, &clusters2);
    println!("{:?}", ri);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_data() -> (Vec<HashSet<u8>>, Vec<HashSet<u8>>) {
        let clusters1 = vec![vec![0u8,8,3, 7], vec![1u8,5], vec![2u8, 4, 6]];
        let clusters1: Vec<HashSet<u8>> = clusters1.iter().map(
            |v| hashset(&v)).collect();
        let clusters2 = vec![vec![0u8,4, 7], vec![1u8,2,3, 6], vec![8u8,5]];
        let clusters2: Vec<HashSet<u8>> = clusters2.iter().map(
            |v| hashset(&v)).collect();
        (clusters1, clusters2)
    }

    #[test]
    fn test_contingency_table() {
        let (clusters1, clusters2) = generate_data();
        let table = contingency_table(&clusters1, &clusters2);
        println!("{:?}", table);
        let table2: Vec<u8> = [2, 1, 1, 0, 1, 1, 1, 2, 0].to_vec();
        let table3 = Array::from_shape_vec((3,3), table2).unwrap();
        assert_eq!(table, table3);
    }

    #[test]
    fn test_matching_elems_count() {
        let (clusters1, clusters2) = generate_data();
        let s1 = &clusters1[0];
        let s2 = &clusters2[0];
        let res = matching_elems_count(&s1, &s2);
        assert_eq!(res, 2);
    }

    #[test]
    fn test_cluster_size_sequence_sqsum() {
        let (clusters1, _) = generate_data();
        let res = cluster_size_sequence_sqsum(&clusters1);
        assert_eq!(res, 29);
    }

    #[test]
    fn test_elements_in_vector() {
        let (clusters1, _) = generate_data();
        let res = elements_in_vectr(&clusters1);
        assert_eq!(res, 9);
    }

    #[test]
    fn test_count_pairwise_cooccurence() {
        let (clusters1, clusters2) = generate_data();
        let res = count_pairwise_cooccurence(&clusters1, &clusters2);
        assert_eq!(res, (2.0, 8.0, 8.0, 18.0));
    }

    #[test]
    fn test_jaccard_index() {
        let (clusters1, clusters2) = generate_data();
        let res = jaccard_index(&clusters1, &clusters2);
        assert_eq!(res, 0.1111111111111111);
    }

    #[test]
    fn test_rand_index() {
        let (clusters1, clusters2) = generate_data();
        let res = rand_index(&clusters1, &clusters2);
        assert_eq!(res, 0.5555555555555556);
    }
}