/// Data gathered from https://www.kaggle.com/vikrishnan/boston-house-prices
/// Boston dataset: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
/// This module shows how to run regression models
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;

#[derive(Debug, Deserialize)]
pub struct BostonHousing {
    crim: f64, // per capita crime rate by town
    zn: f64, // proportion of residential land zoned for lots over 25,000 sq.ft.
    indus: f64, // proportion of non-retail business acres per town.
    chas: f64, // Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    nox: f64, // nitrogen oxides concentration (parts per 10 million).
    rm: f64, // average number of rooms per dwelling.
    age: f64, // proportion of owner-occupied units built prior to 1940.
    dis: f64, // weighted mean of distances to five Boston employment centres.
    rad: f64, // index of accessibility to radial highways.
    tax: f64, // full-value property-tax rate per $10,000.
    ptratio: f64, // pupil-teacher ratio by town.
    black: f64, // 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
    lstat: f64, // lower status of the population (percent).
    medv: f64, // median value of owner-occupied homes in $1000s.
}

impl BostonHousing {
    pub fn new(v: Vec<&str>) -> BostonHousing {
        let f64_formatted: Vec<f64> = v.iter().map(|s| s.parse().unwrap()).collect();
        BostonHousing { crim: f64_formatted[0], zn: f64_formatted[1], indus: f64_formatted[2], chas: f64_formatted[3],
                        nox: f64_formatted[4], rm: f64_formatted[5], age: f64_formatted[6], dis: f64_formatted[7],
                        rad: f64_formatted[8], tax: f64_formatted[9], ptratio: f64_formatted[10], black: f64_formatted[11],
                        lstat: f64_formatted[12], medv: f64_formatted[13] }
    }

    pub fn into_feature_vector(&self) -> Vec<f64> {
        vec![self.crim, self.zn, self.indus, self.chas, self.nox,
             self.rm, self.age, self.dis, self.rad,
             self.tax, self.ptratio, self.black, self.lstat]
    }

    pub fn into_targets(&self) -> f64 {
        self.medv
    }
}

fn get_boston_record(s: String) -> BostonHousing {
    let v: Vec<&str> = s.split_whitespace().collect();
    let b: BostonHousing = BostonHousing::new(v);
    b
}

pub fn get_boston_records_from_file(filename: impl AsRef<Path>) -> Vec<BostonHousing> {
    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines().enumerate()
        .map(|(n, l)| l.expect(&format!("Could not parse line no {}", n)))
        .map(|r| get_boston_record(r))
        .collect()
}

#[derive(Debug, Deserialize)]
pub struct Flower {
    sepal_length: f32, // everything needs to be f32, other types wont do in rusty machine
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    species: String,
}

impl Flower {
    pub fn into_feature_vector(&self) -> Vec<f32> {
        vec![self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]
    }

    pub fn into_labels(&self) -> f32 {
        match self.species.as_str() {
            "setosa" => 0.,
            "versicolor" => 1.,
            "virginica" => 2.,
            l => panic!("Not able to parse the target. Some other target got passed. {:?}", l),
        }
    }

    pub fn into_int_labels(&self) -> u64 {
        match self.species.as_str() {
            "setosa" => 0,
            "versicolor" => 1,
            "virginica" => 2,
            l => panic!("Not able to parse the target. Some other target got passed. {:?}", l),
        }
    }
}

pub struct SpookyAuthor {
    id: String,
    text: String,
    author: String
}

#[cfg(test)]
mod tests {
    use super::*;
    use csv;
    use std::io;


    #[test]
    fn test_flower() {
        let data = "sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n";
        let mut rdr = csv::Reader::from_reader(data.as_bytes());
        let mut data = Vec::new();
        for result in rdr.deserialize() {
            let r: Flower = result.unwrap();
            data.push(r); // data contains all the records
        }
        assert_eq!(data[0].sepal_length, 5.1);
    }

    #[test]
    fn test_flower_data_f32() {
        let data = "sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n";
        let mut rdr = csv::Reader::from_reader(data.as_bytes());
        let mut data = Vec::new();
        for result in rdr.deserialize() {
            let r: Flower = result.unwrap();
            data.push(r); // data contains all the records
        }
        let x: f32 = data[0].sepal_length;
        assert_eq!(x, 5.1);

    }

    #[test]
    fn test_into_feature_vector() {
        let data = "sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n";
        let mut rdr = csv::Reader::from_reader(data.as_bytes());
        let mut data = Vec::new();
        for result in rdr.deserialize() {
            let r: Flower = result.unwrap();
            data.push(r); // data contains all the records
        }

        let flower_x_train: Vec<f32> = data.iter().flat_map(|r| r.into_feature_vector()).collect();
        assert_eq!(flower_x_train, [5.1, 3.5, 5.1, 0.2]);
    }

    #[test]
    fn test_into_labels() {
        let data = "sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n";
        let mut rdr = csv::Reader::from_reader(data.as_bytes());
        let mut data = Vec::new();
        for result in rdr.deserialize() {
            let r: Flower = result.unwrap();
            data.push(r); // data contains all the records
        }

        let flower_x_train: Vec<f32> = data.iter().map(|r| r.into_labels()).collect();
        assert_eq!(flower_x_train, [0.0]);
    }
}
