use bincode::{serialize, deserialize};
use std::fs::File;
use std::io::prelude::*;
use ndarray::prelude::*;

// save nn to file
pub fn save(path: &str, data: &Array2<f32>) {        
    // convert simplified nn to Vec<u8>
    let encoded: Vec<u8> = serialize(
        data
    ).unwrap();
 
    // open file and write whole Vec<u8>
    let mut file = File::create(path).unwrap();
    file.write_all(&encoded).unwrap();
} 
 
// load nn from file
pub fn load(path: &str) -> Array2<f32> {
    // convert readed Vec<u8> to plain nn
    let mut buffer = vec![];
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    let decoded: Array2<f32> = deserialize(&buffer).unwrap();
 
    decoded
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
