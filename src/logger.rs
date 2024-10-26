#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use bincode::{serialize, deserialize};
use std::fs::{File,self};
use std::io::prelude::*;
use toml;
use ndarray::prelude::*;
use rayon::prelude::*;
use unique_id::{Generator, string::StringGenerator};
use macroquad::prelude::*;

use crate::lenia::{Channel, Eco, Function, Layer};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerData {
    kernel: Function,
    growth_map: Function,
    matrix_id: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatrixData {
    uid: String,
    matrix: Array2<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TomlData {
    uid: String,
    delta: f32,
    size: (usize, usize),
    cycles: usize,
    fitness: f32,
    layer: Vec<LayerData>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstanceData {
    uid: String,
    nick: String,
    toml: String,
    matrix: Vec<String>,
    active: bool
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TomlCorrelations {
    correlation: Vec<InstanceData>
}


#[derive(Debug)]
pub struct Logger {
    tomls: Vec<TomlData>,
    matrices: Vec<MatrixData>,
    correlations: TomlCorrelations,
    preview_slice: Vec<u8>, // for previews
    visu_slice: Vec<u8>, // for visualization, so updated amost every frame
    buffer_slice: Vec<u8>, // buffer, may save some time needed for memory reservation
    gen: StringGenerator,
}

impl Logger {
    pub fn new() -> Self {
        Self { tomls: vec![], matrices: vec![], preview_slice: vec![], visu_slice: vec![], buffer_slice: vec![],
        correlations: TomlCorrelations {correlation: vec![]}, gen: StringGenerator::default() }
    }
    
    pub fn image(&mut self, matrix: &Array2<f32>) -> ( (usize, usize), &Vec<u8>) {
        self.buffer_slice.resize_with(matrix.len() * 4, || {0});
        let size = ( matrix.len_of(Axis(0)), matrix.len_of(Axis(1)) );
        self.buffer_slice.par_chunks_mut(4).enumerate().for_each(|(i, x)|{
            let col = (matrix[[i%size.0, i/size.0]] * 255.0) as isize;
            
            x[0] = (-(col/4 - 16).pow(2) + 255).clamp(0, 255) as u8; // 3-16
            x[1] = (-(col/4 - 32).pow(2) + 255).clamp(0, 255) as u8; // 3-44
            x[2] = (-(col/4 - 48).pow(2) + 255).clamp(0, 255) as u8; // 3-72
            //x[3] = (col*2).clamp(0, 255) as u8;
            if col > 0 {x[3] = 255;} else {x[3] = 0;}
        });
        ( size, &self.buffer_slice)
    }

    // returns pushed instance index
    // translates lenia simple id to uid via slice
    fn push_toml(&mut self, eco: &Eco, matrix_uids: &[String]) -> &String {
        let toml;
        let mut layer_data = vec![];
        eco.layers.iter().for_each(|l|{
            layer_data.push(LayerData {
                kernel: l.kernel.clone(), growth_map: l.growth_map.clone(), matrix_id: matrix_uids[l.channel_id].clone()
            });
        });
        let uid = self.gen.next_id();
        toml = TomlData {
            delta: eco.delta, size: eco.size, cycles: eco.cycles, fitness: eco.fitness, layer: layer_data, uid: uid.clone()
        };
        self.tomls.push(toml);
        &self.tomls.last().unwrap().uid
    }
    // returns matrix index
    pub fn push_matrix(&mut self, m: &Array2<f32>) -> &String {
        let uid =  self.gen.next_id();
        self.matrices.push( MatrixData { uid: uid.clone(), matrix: m.clone() } );
        &self.matrices.last().unwrap().uid
    }

    // uses eco reference to correlate to toml in list
    pub fn push_correlation(&mut self, eco: &Eco) -> &String {
        let mut matrix = vec![];
        eco.channels.iter().for_each(|ch|{
            matrix.push(self.push_matrix(&ch.matrix).to_string());
        });
        let uid = self.gen.next_id();
        let instance = InstanceData {
            uid,
            nick: "idk".to_string(),
            toml: self.push_toml(eco, &matrix).to_string(),
            matrix, active: true
        };
        self.correlations.correlation.push(instance);
        &self.correlations.correlation.last().unwrap().uid
    }

    pub fn get_correlation_list(&self) -> Vec<(&String, &String)> {
        let mut list = vec![];
        self.correlations.correlation.iter().for_each(|c|{
            list.push((&c.uid, &c.nick));
        });
        list
    }

    // Get eco from index, if index doesn't exists, returns None
    pub fn get_correlation(&self, uid: &String) -> Option<Eco> {
        let real_idx = match self.correlations.correlation.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return None;}
        };
        let instance = self.correlations.correlation.get(real_idx).unwrap();
        
        let real_idx = match self.tomls.iter().position(|c| c.uid == instance.toml) {
            Some(c) => c,
            None => {return None;}
        };
        let toml;
        match self.tomls.get(real_idx) {
            Some(t) => {toml = t.clone();}
            None => return None
        }
        
        // in foreach it's hard to return none to parent
        let mut channels = vec![];
        let mut matrix_hashmap = vec![]; // if is in order (should be) index is channel number
        for i in instance.matrix.clone() {
            let real_idx = match self.matrices.iter().position(|c| c.uid == i) {
                Some(c) => c,
                None => {return None;}
            };
            match self.matrices.get(real_idx) {
                Some(m) => {channels.push(
                    Channel::new(m.matrix.clone()));
                    matrix_hashmap.push(m.uid.clone());
                }
                None => return None
            };
        }
        
        let mut layers = vec![];
        toml.layer.iter().for_each(|l|{
            let id = matrix_hashmap.iter().position(|m| *m == l.matrix_id).unwrap();
            layers.push(Layer::new(l.kernel.clone(), l.growth_map.clone(), id));
        });
        Some(Eco::new(toml.size, toml.delta, toml.cycles, channels, layers))
    }

    // Return false if there is no instance at index
    fn update_toml(&mut self, uid: &String, toml: TomlData) -> bool {
        let real_idx = match self.tomls.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return false;}
        };
        self.tomls[real_idx] = toml;
        true
    }
    fn update_matrix(&mut self, uid: &String, matrix: Array2<f32>) -> bool {
        let real_idx = match self.matrices.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return false;}
        };
        self.matrices[real_idx].matrix = matrix;
        true
    }
    // update (toml, matrices)
    pub fn update_correlation(&mut self, uid: &String, eco: &Eco, option: (bool, bool)) -> bool {
        let real_idx = match self.correlations.correlation.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return false;}
        };
        self.push_correlation(eco);
        //self.correlations.correlation[idx] = self.correlations.correlation.last().unwrap().clone();
        if option.0 {
            self.correlations.correlation[real_idx].toml = self.correlations.correlation.last().unwrap().clone().toml;
        }
        if option.1 {
            self.correlations.correlation[real_idx].matrix = self.correlations.correlation.last().unwrap().clone().matrix;
        }
        self.correlations.correlation.pop();
        true
    }

    // removing element shifts rest to the left, that breaks instances, thus the need for shifting
    // If correlation have missing elements it's marked as inactive
    // removing instance breaks indexing, so they need to by referenced by id
    pub fn pop_toml(&mut self, uid: &String) -> bool {
        let real_idx = match self.tomls.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return false;}
        };
        self.tomls.remove(real_idx);
        self.correlations.correlation.iter_mut().for_each(|c|{
            if c.toml == *uid {c.active = false;}
        });
        true
    }
    pub fn pop_matrix(&mut self, uid: &String) -> bool {
        let real_idx = match self.matrices.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return false;}
        };
        self.matrices.remove(real_idx);
        self.correlations.correlation.iter_mut().for_each(|c|{
            c.matrix.iter_mut().for_each(|m|{
                if *m == *uid {c.active = false;}
            });
        });
        true
    }
    pub fn pop_correlation(&mut self, uid: &String) -> bool {
        let real_idx = match self.correlations.correlation.iter().position(|c| c.uid == *uid) {
            Some(c) => c,
            None => {return false;}
        };
        self.correlations.correlation.remove(real_idx);
        true
    }

    // overwrites old saves, so it's recommended to load before saving, returns amount of records
    pub fn save_to_file(&mut self) -> (usize, usize, usize) {
        let _ = fs::remove_dir_all("data");
        let path = "data".to_owned();
        fs::create_dir_all(path.clone()).unwrap();
        {
            let toml = toml::to_string(&self.correlations).unwrap();
            let mut file = File::create(path.clone() + "/correlations.toml").unwrap();
            file.write(toml.as_bytes()).unwrap();
        }

        let path = "data/toml".to_owned();
        fs::create_dir_all(path.clone()).unwrap();
        self.tomls.iter().for_each(|t|{
            let toml = toml::to_string(&t).unwrap();
            let mut file = File::create(path.clone() + "/" + &t.uid.to_string() + ".toml").unwrap();
            file.write(toml.as_bytes()).unwrap();
        });
        let path = "data/matrix".to_owned();
        fs::create_dir_all(path.clone()).unwrap();
        self.matrices.clone().iter().for_each(|m|{
            let rtx = self.image(&m.matrix.clone());
            let tx = Texture2D::from_rgba8(rtx.0.0 as u16, rtx.0.1 as u16, rtx.1);
            let img = tx.get_texture_data();
            img.export_png(&(path.clone() + "/" + &m.uid.to_string() + ".png"));
            
            let serialized: Vec<u8> = serialize(m).unwrap();
            let mut file = File::create(path.clone() + "/" + &m.uid.to_string() + ".bin").unwrap();
            file.write_all(&serialized).unwrap();
        });
        (self.correlations.correlation.len(), self.tomls.len(), self.matrices.len())
    }

    // returns amount of loaded instances
    pub fn load_from_file(&mut self) -> (usize, usize, usize) {
        self.tomls.clear();
        let dir = match fs::read_dir("data/toml") {
            Ok(d) => d,
            Err(_) => return (0,0,0),
        };
        let mut entries = dir.map(|res| res.map(|e| e.path())).collect::<Result<Vec<_>, std::io::Error>>().unwrap();
        entries.sort_unstable();
        let entries = entries.iter().filter(|e| e.is_file() && e.file_name().unwrap().to_string_lossy().contains(".toml") ).collect::<Vec<_>>();
        entries.iter().for_each(|e|{
            let toml_raw = fs::read_to_string(e).unwrap();
            let toml_data:TomlData = toml::from_str(&toml_raw).unwrap();
            self.tomls.push(toml_data);
        });

        self.matrices.clear();
        let dir = match fs::read_dir("data/matrix") {
            Ok(d) => d,
            Err(_) => return (0,self.tomls.len(),0),
        };
        let mut entries = dir.map(|res| res.map(|e| e.path())).collect::<Result<Vec<_>, std::io::Error>>().unwrap();
        entries.sort_unstable();
        let entries = entries.iter().filter(|e| e.is_file() && e.file_name().unwrap().to_string_lossy().contains(".bin") ).collect::<Vec<_>>();
        entries.iter().for_each(|e|{
            let mut buffer = vec![];
            let mut file = File::open(e).unwrap();
            file.read_to_end(&mut buffer).unwrap();
            let deserialized: MatrixData = deserialize(&buffer).unwrap();
            self.matrices.push(deserialized);
        });

        self.correlations.correlation.clear();
        let toml_raw = match fs::read_to_string("data/correlations.toml") {
            Ok(d) => d,
            Err(_) => return (0,self.tomls.len(),self.matrices.len()),
        };
        let correlation_data: TomlCorrelations = toml::from_str(&toml_raw).unwrap();
        self.correlations = correlation_data;
        (self.correlations.correlation.len(), self.tomls.len(), self.matrices.len())
    }
}

