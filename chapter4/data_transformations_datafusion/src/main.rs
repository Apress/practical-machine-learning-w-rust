use std::sync::Arc;

use arrow;
use datafusion;
use arrow::array::{BinaryArray, Float64Array, UInt16Array, ListArray};
use arrow::datatypes::{DataType, Field, Schema};

use datafusion::execution::context::ExecutionContext;

fn main() {
    // create local execution context
    let mut ctx = ExecutionContext::new();

    // define schema for data source (csv file)
    let schema = Arc::new(Schema::new(vec![
        Field::new("PassengerId", DataType::Int32, false),
        Field::new("Survived", DataType::Int32, false),
        Field::new("Pclass", DataType::Int32, false),
        Field::new("Name", DataType::Utf8, false),
        Field::new("Sex", DataType::Utf8, false),
        Field::new("Age", DataType::Int32, true),
        Field::new("SibSp", DataType::Int32, false),
        Field::new("Parch", DataType::Int32, false),
        Field::new("Ticket", DataType::Utf8, false),
        Field::new("Fare", DataType::Float64, false),
        Field::new("Cabin", DataType::Utf8, true),
        Field::new("Embarked", DataType::Utf8, false),
    ]));

    // register csv file with the execution context
    ctx.register_csv(
        "titanic",
        "titanic/train.csv",
        &schema,
        true,
    );

    // simple projection and selection
    let sql = "SELECT Name, Sex FROM titanic WHERE Fare > 8";
    let sql1 = "SELECT MAX(Fare) FROM titanic WHERE Survived = 1";

    // execute the query
    let relation = ctx.sql(&sql, 1024 * 1024).unwrap();
    let relation1 = ctx.sql(&sql1, 1024 * 1024).unwrap();

    // display the relation
    let mut results = relation.borrow_mut();
    let mut results1 = relation1.borrow_mut();

    while let Some(batch) = results.next().unwrap() {
        println!(
            "RecordBatch has {} rows and {} columns",
            batch.num_rows(),
            batch.num_columns()
        );

        let name = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        let sex = batch
            .column(1)
            .as_any()
            // .downcast_ref::<Float64Array>()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let name_value: String = String::from_utf8(name.value(i).to_vec()).unwrap();
            let sex_value: String = String::from_utf8(sex.value(i).to_vec()).unwrap();

            println!("name: {}, sex: {}", name_value, sex_value,);
        }
    }
    while let Some(batch) = results1.next().unwrap() {
        println!(
            "RecordBatch has {} rows and {} columns",
            batch.num_rows(),
            batch.num_columns()
        );

        let name = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let name_value: f64 = name.value(i);

            println!("name: {}", name_value,);
        }
    }
    println!("Hello, world!");
}