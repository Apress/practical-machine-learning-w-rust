use std::path::PathBuf;
use std::error::Error;

use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::{Session, SessionOptions, SessionRunArgs, Tensor};
use structopt::StructOpt;
use image;
use image::GenericImageView;
use image::Rgba;
use imageproc;
use imageproc::rect::Rect;
use imageproc::drawing::draw_hollow_rect_mut;

const LINE_COLOUR: Rgba<u8> = Rgba {
    data: [0, 255, 0, 0],
};

#[derive(Debug, StructOpt)]
#[structopt(name = "face-detection-tf", about = "Face Identification")]
struct Opt {
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: PathBuf,

    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: PathBuf
}

#[derive(Copy, Clone, Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

/// read image, convert to RGB and load to a tensor
/// for face prediction.
fn get_input_image_tensor(opt: &Opt) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_image = image::open(&opt.input)?;
    
    let mut flattened: Vec<f32> = Vec::new();
    for (_x, _y, rgb) in input_image.pixels() {
        flattened.push(rgb[2] as f32);
        flattened.push(rgb[1] as f32);
        flattened.push(rgb[0] as f32);
    }
    let input = Tensor::new(
        &[input_image.height() as u64, input_image.width() as u64, 3])
        .with_values(&flattened)?;
    Ok(input)
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    println!("{:?}", (opt.input.to_owned(), opt.output.to_owned()));
    let input = get_input_image_tensor(&opt)?;

    //First, we load up the graph as a byte array
    let model = include_bytes!("../mtcnn.pb");

    //Then we create a tensorflow graph from the model
    let mut graph = Graph::new();
    graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

    let session = Session::new(&SessionOptions::new(), &graph)?;
    let min_size = Tensor::new(&[]).with_values(&[40f32])?;
    let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
    let factor = Tensor::new(&[]).with_values(&[0.709f32])?;

    let mut args = SessionRunArgs::new();

    //Load our parameters for the model
    args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
    args.add_feed(&graph.operation_by_name_required("thresholds")?, 0, &thresholds);
    args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);

    //Load our input image
    args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);

    let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
    let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);

    session.run(&mut args)?;

    let bbox_res: Tensor<f32> = args.fetch(bbox)?;
    let prob_res: Tensor<f32> = args.fetch(prob)?;

    println!("{:?}", bbox_res.dims()); // [120, 4]
    println!("{:?}", prob_res.dims()); // [120]

    //Let's store the results as a Vec<BBox>
    let bboxes: Vec<_> = bbox_res
        .chunks_exact(4) // Split into chunks of 4
        .zip(prob_res.iter()) // Combine it with prob_res
        .map(|(bbox, &prob)| BBox {
            y1: bbox[0],
            x1: bbox[1],
            y2: bbox[2],
            x2: bbox[3],
            prob,
        }).collect();
    println!("BBox Length: {}, Bboxes:{:#?}", bboxes.len(), bboxes);

    let mut output_image = image::open(&opt.input)?;

    for bbox in bboxes {
        let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
        draw_hollow_rect_mut(&mut output_image, rect, LINE_COLOUR);
    }
    output_image.save(&opt.output)?;

    Ok(())
}