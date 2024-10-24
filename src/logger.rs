#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use bincode::{serialize, deserialize};
use std::fs::{File,self};
use std::io::prelude::*;
use toml;
use ndarray::prelude::*;

use crate::lenia::{Channel, Eco, Function, Layer};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerData {
    kernel: Function,
    growth_map: Function,
    channel_id: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TomlData {
    delta: f32,
    size: (usize, usize),
    cycles: usize,
    fitness: f32,
    layer: Vec<LayerData>,
}

#[derive(Debug)]
pub struct Logger {
    instances: Vec<(TomlData, Vec<Array2<f32>>)> // parameters and matrixes
}

impl Logger {
    pub fn new() -> Self {
        Self { instances: vec![] }
    }

    // returns pushed instance index
    pub fn push_instance(&mut self, eco: Eco) -> usize {
        let mut matrix = vec![];
        eco.channels.iter().for_each(|ch|{
            matrix.push(ch.matrix.clone());
        });
        let toml;
        let mut layer_data = vec![];
        eco.layers.iter().for_each(|l|{
            layer_data.push(LayerData {
                kernel: l.kernel.clone(), growth_map: l.growth_map.clone(), channel_id: l.channel_id
            });
        });
        toml = TomlData {
            delta: eco.delta, size: eco.size, cycles: eco.cycles, fitness: eco.fitness, layer: layer_data
        };
        self.instances.push((toml, matrix));
        self.instances.len() - 1
    }

    // Get eco from index, if index doesn't exists, returns None
    pub fn get_instance(&self, idx: usize) -> Option<Eco> {
        match self.instances.get(idx) {
            Some(_) => {}
            None => return None
        }
        let toml = &self.instances[idx].0;
        let matrix = &self.instances[idx].1;
        let mut channels = vec![];
        matrix.iter().for_each(|m|{
            channels.push(Channel::new(m.clone()));
        });
        let mut layers = vec![];
        toml.layer.iter().for_each(|l|{
            layers.push(Layer::new(l.kernel.clone(), l.growth_map.clone(), l.channel_id));
        });
        Some(Eco::new(toml.size, toml.delta, toml.cycles, channels, layers))
    }

    // Return false if there is no instance at index
    pub fn update_instance(&mut self, idx: usize, eco: Eco) -> bool {
        if idx >= self.instances.len() {return false;}
        self.push_instance(eco);
        self.instances[idx] = self.instances.last().unwrap().clone();
        self.instances.pop();
        true
    }

    // Return false if there is no instance at index
    pub fn pop_instance(&mut self, idx: usize) -> bool {
        if idx >= self.instances.len() {return false;}
        self.instances.remove(idx);
        true
    }

    // overwrites old saves, so it's recommended to load before saving, returns amount of records
    pub fn save_to_file(&self) -> usize {
        let _ = fs::remove_dir_all("data");
        self.instances.iter().enumerate().for_each(|(i, s)|{
            let path = "data/save".to_owned() + &i.to_string();
            fs::create_dir_all(path.clone()).unwrap();
            s.1.iter().enumerate().for_each(|(j, x)|{
                let serialized: Vec<u8> = serialize(x).unwrap();
                let mut file = File::create(path.clone() + &("/matrix".to_owned() + &j.to_string() + ".bin")).unwrap();
                file.write_all(&serialized).unwrap();
            });

            let toml = toml::to_string(&s.0).unwrap();
            let mut file = File::create(path + "/toml.toml").unwrap();
            file.write(toml.as_bytes()).unwrap();
        });
        self.instances.len()
    }

    // returns amount of loaded instances
    pub fn load_from_file(&mut self) -> usize {
        self.instances.clear();
        let dir = match fs::read_dir("data/") {
            Ok(d) => d,
            Err(_) => return 0,
        };
        let mut entries = dir.map(|res| res.map(|e| e.path())).collect::<Result<Vec<_>, std::io::Error>>().unwrap();
        entries.sort_unstable();
        let entries = entries.iter().filter(|e| e.is_dir() && e.file_name().unwrap().to_string_lossy().contains("save") ).collect::<Vec<_>>();
        entries.iter().for_each(|e|{
            let mut t = e.to_owned().to_owned();
            t.push("toml.toml");
            let toml_raw = fs::read_to_string(&t).unwrap();
            let toml_data:TomlData = toml::from_str(&toml_raw).unwrap();

            let mut matrix: Vec<Array2<f32>> = vec![];
            let mut m_dir = fs::read_dir(e).unwrap()
                .map(|res| res.map(|e| e.path()))
                .collect::<Result<Vec<_>, std::io::Error>>().unwrap();
            m_dir.sort_unstable();
            let m_dir = m_dir.iter().filter(|m| m.is_file() && m.file_name().unwrap().to_string_lossy().contains("matrix") ).collect::<Vec<_>>();
            m_dir.iter().for_each(|m|{
                let mut buffer = vec![];
                let mut file = File::open(m.to_owned().to_owned()).unwrap();
                file.read_to_end(&mut buffer).unwrap();
                let deserialized: Array2<f32> = deserialize(&buffer).unwrap();
                matrix.push(deserialized);
            });
            self.instances.push((toml_data, matrix));
        });
        self.instances.len()
    }
}

