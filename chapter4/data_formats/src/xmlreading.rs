use std::fs::File;
use std::result::Result;
use std::error::Error;

use serde_xml_rs;
use serde_xml_rs::from_reader;

#[derive(Deserialize, Debug)]
struct Project {
    name: String,
    libraries: Vec<Libraries>,
    module: Vec<Module>,
}

#[derive(Deserialize, Debug)]
struct Module {
    files: Vec<Files>,
    #[serde(default)]
    libraries: Vec<Libraries>,
}

#[derive(Deserialize, Debug)]
struct Files {
    file: Vec<FileName>,
}

#[derive(Deserialize, Debug)]
struct FileName {
    name: String,
    #[serde(rename = "type")]
    lang: String,
    #[serde(rename = "$value")]
    body: String,
}

#[derive(Deserialize, Debug)]
struct Libraries {
    library: Vec<Library>,
}

#[derive(Deserialize, Debug)]
struct Library {
    #[serde(rename = "groupId")]
    group_id: String,
    #[serde(rename = "artifactId")]
    artifact_id: String,
    version: String,
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let file = File::open("data/sample_2.xml").unwrap();
    let project: Project = from_reader(file).unwrap();
    println!("{:#?}", project.libraries[0].library[0]);
    println!("{:#?}", project);
    Ok(())
}
