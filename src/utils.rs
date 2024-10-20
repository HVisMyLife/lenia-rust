use bincode::{serialize, deserialize};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::prelude::*;
use toml;
use ndarray::prelude::*;

#[derive(Deserialize, Serialize)]
struct FileData {
    params: [Params;2]
}

#[derive(Deserialize, Serialize)]
struct Params {
    size: i32,
    delta: f32,
    offset: f32,
    width: f32
}

// save nn to file
pub fn save(path: &str, data: ( &Array2<f32>, [(i32, (f32, f32, f32)); 2] )  ) {        
    // matrix data save
    let encoded: Vec<u8> = serialize(
        data.0
    ).unwrap();
    let mut file = File::create(path.to_owned() + ".bin").unwrap();
    file.write_all(&encoded).unwrap();

    // params save
    let file_data = FileData {
        params: 
        [Params {
            size: data.1[0].0,
            delta: data.1[0].1.0,
            offset: data.1[0].1.1,
            width: data.1[0].1.2,
        },
        Params {
            size: data.1[1].0,
            delta: data.1[1].1.0,
            offset: data.1[1].1.1,
            width: data.1[1].1.2,
        }]
    };
    let toml = toml::to_string(&file_data).unwrap();
    let mut file = File::create(path.to_owned() + ".toml").unwrap();
    file.write(toml.as_bytes()).unwrap();
} 
 
// load nn from file
pub fn load(path: &str) -> ( Array2<f32>, [(i32, (f32, f32, f32)); 2] ) {
    // convert readed Vec<u8> to plain nn
    let mut buffer = vec![];
    let mut file = File::open(path.to_owned() + ".bin").unwrap();
    file.read_to_end(&mut buffer).unwrap();
    let decoded: Array2<f32> = deserialize(&buffer).unwrap();
 
    let file_data = fs::read_to_string(path.to_owned() + ".toml").unwrap();

    let toml:FileData = toml::from_str(&file_data).unwrap();
    (
        decoded, 
        [
            (
                toml.params[0].size,
                (toml.params[0].delta, toml.params[0].offset, toml.params[0].width )
            ),
            (
                toml.params[1].size,
                (toml.params[1].delta, toml.params[1].offset, toml.params[1].width )
            ),
        ]
    )
} 

pub struct FrameTimeAnalyzer {
    frame: Vec<f32>,
    s_time: f32,
}

impl FrameTimeAnalyzer {
    pub fn new(length: usize) -> Self {
        FrameTimeAnalyzer {
            frame: vec![0.; length],
            s_time: 0.,
        }
    }

    pub fn add_frame_time(&mut self, time: f32) {
        self.frame.pop();
        self.frame.insert(0, time);
    }

    pub fn smooth_frame_time(&mut self) -> &f32 {
        self.s_time = self.frame.iter().sum::<f32>() / (self.frame.len() as f32);
        &self.s_time
    }
}
