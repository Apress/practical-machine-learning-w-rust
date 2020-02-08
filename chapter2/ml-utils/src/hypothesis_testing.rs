use std::vec::Vec;
use std::collections::HashMap;
// use std::io::Error;
use std::error::Error;

use rand;
use rand::distributions::{Bernoulli, Distribution};

#[derive(Debug, PartialEq, Eq, Hash)]
enum User {
    Group,
    Converted,
}

fn generate_data(control_size: u32, test_size: u32, p_control: f64, p_test: f64) -> Vec<HashMap<User, bool>> {
    // initiate empty container.
    let mut data = vec![];

    let total = control_size + test_size;
    
    let group_bern = Bernoulli::new(0.5); // we need to divide the whole population equally

    let control_bern = Bernoulli::new(p_control);
    let test_bern = Bernoulli::new(p_test);

    for _ in 0..total {

        let mut row = HashMap::new();
        let v = group_bern.sample(&mut rand::thread_rng());
        row.insert(User::Group, v);
        
        let converted_v = match v {
            // true means control and false means test
            true => control_bern.sample(&mut rand::thread_rng()),
            false => test_bern.sample(&mut rand::thread_rng()),
        };
        row.insert(User::Converted, converted_v);
        data.push(row);
    }
    data
}

fn find_rate_difference(data: &Vec<HashMap<User, bool>>) -> Result<f64, Box<Error>> {
    let mut total_control_groups: usize = 0;
    let mut converted_control_group: usize = 0;
    let mut converted_test_group: usize = 0;
    for d in data {
        let user_group = d.get(&User::Group)
            .expect("data must have group and converted");
        let user_conversion = d.get(&User::Converted)
            .expect("data must have group and converted");
        if user_group == &true {
            total_control_groups += 1;
            if user_conversion == &true {
                converted_control_group += 1;
            }
        } else {
            if user_conversion == &true {
                converted_test_group += 1;
            }
        }
    }
    let total_test_group = data.len() - total_control_groups;
    let control_rate = converted_control_group as f64/total_control_groups as f64;
    let test_rate = converted_test_group as f64/total_test_group as f64;
    Ok(test_rate - control_rate)
}

fn main() {
    // A is control and B is test
    let control_size = 1000;
    let test_size = 1000;

    let bcr = 0.10;  // baseline conversion rate
    let d_hat = 0.02;  // difference between the groups
    let data = generate_data(control_size, test_size, bcr, bcr + d_hat); // we want data that is a little better than baseline.
    println!("{:?}", data);

    let x = find_rate_difference(&data);
    println!("{:?}", x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_data() {
        let data = generate_data(10, 10, 0.1, 0.02);
        assert_eq!(data.len(), 20);
        assert_eq!(data[0].contains_key(&User::Group), true);
    }

    #[test]
    fn test_find_rate_difference() {
        let mut data = vec![];
        let data1: HashMap<_, _> = vec![(User::Group, false), (User::Converted, false)].into_iter().collect();
        data.push(data1);
        let data2: HashMap<_, _> = vec![(User::Group, true), (User::Converted, true)].into_iter().collect();
        data.push(data2);
        let res = find_rate_difference(&data).unwrap();
        assert_eq!(res, -1.0);
    }
}