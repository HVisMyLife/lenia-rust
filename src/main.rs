//#![allow(clippy::ptr_arg)]
use macroquad::prelude::*;
use ndarray::prelude::*;

mod utils;
use serde::{Deserialize, Serialize};
use bincode::{serialize, deserialize};
use std::fs::{File,self};
use std::io::prelude::*;
use toml;

use utils::FrameTimeAnalyzer;

mod lenia;
use lenia::{Eco, Function};

// TODO: loading save into ecosystem, ui value modification, ui fitness graph, tuning AI


// size of map for convolution, it is later padded for warp func, so we have to cut it for now
pub const MAP_SIZE: (usize, usize) = (1280, 820);
pub const SEED: u64 = 1;

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(SEED);
    let font = macroquad::text::load_ttf_font("font.ttf").await.unwrap();

    let mut eco = Eco::new((MAP_SIZE.0, MAP_SIZE.1), vec![], vec![]);
    eco.init();

    // LOGGER
    let mut logger = Logger::new();
    let mut layer_data: Vec<LayerData> = vec![];
    eco.layers.iter().for_each(|l|{
        layer_data.push(LayerData {
            kernel: l.kernel.clone(), growth_map: l.growth_map.clone(), channel_id: l.channel_id
        });
    });
    let toml_data = TomlData {
        delta: eco.delta, size: MAP_SIZE, cycles: eco.cycles, fitness: eco.fitness, layer: layer_data
    };
    let mut matrix_data: Vec<Array2<f32>> = vec![];
    eco.channels.iter().for_each(|ch|{
        matrix_data.push(ch.matrix.clone());
    });
    logger.load();
    logger.push_instance(toml_data, matrix_data);
    logger.save();
    // LOGGER

    let mut ui = UI::new(font);

    loop {
        clear_background(Color::from_rgba(24, 24, 24, 255));
        if !ui.pause {
            eco.evaluate();
        }
        
        let rtx = eco.image();       
        let tx = Texture2D::from_rgba8(rtx.0.0 as u16, rtx.0.1 as u16, rtx.1);
        draw_texture(&tx, 0., 0., WHITE);

        if ui.execute() {break};
        next_frame().await
    }
}



#[derive(Debug, Serialize, Deserialize)]
struct LayerData {
    kernel: Function,
    growth_map: Function,
    channel_id: usize,
}
#[derive(Debug, Serialize, Deserialize)]
struct TomlData {
    delta: f32,
    size: (usize, usize),
    cycles: usize,
    fitness: f32,
    layer: Vec<LayerData>,
}

#[derive(Debug)]
struct Logger {
    instances: Vec<(TomlData, Vec<Array2<f32>>)> // parameters and matrixes
}

impl Logger {
    fn new() -> Self {
        Self { instances: vec![] }
    }

    // returns pushed instance index
    fn push_instance(&mut self, toml: TomlData, matrix: Vec<Array2<f32>>) -> usize {
        self.instances.push((toml, matrix));
        self.instances.len() - 1
    }

    fn get_instance(&self, idx: usize) -> &(TomlData, Vec<Array2<f32>>) {
        &self.instances[idx]
    }

    // overwrites old saves, so it's recommended to load before saving
    fn save(&self) {
        fs::remove_dir_all("data").unwrap();
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
    }

    fn load(&mut self) {
        self.instances.clear();
        let mut entries = fs::read_dir("data/").unwrap()
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, std::io::Error>>().unwrap();
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
    }
}



struct UI {
    fta: FrameTimeAnalyzer,
    pub pause: bool,
    //autotune: bool,
    font: Font,
}

impl UI {
    fn new(font: Font) -> Self {

        UI { 
            fta: FrameTimeAnalyzer::new(16), 
            pause: false, 
            font,
        }
    }

    fn execute(&mut self) -> bool {
        if is_key_pressed(KeyCode::A) {self.pause = !self.pause;}
        let mut tp = TextParams { 
            font: Some(&self.font), 
            font_size: 30, 
            font_scale: 1., 
            font_scale_aspect: 1., 
            rotation: 0., 
            color: WHITE 
        };
        self.fta.add_frame_time(get_frame_time()*1000.);
        draw_text_ex(&(self.fta.smooth_frame_time().round().to_string() + " ms"), 8., 30., tp.clone());
        
        //draw_text_ex(&("deltaT0: ".to_owned() + &eco.layer[0].g_params.0.to_string()), 8., 30.*2., tp.clone());
        
        tp.font_size = 128;
        if self.pause {draw_text_ex("PAUSE", 256., 256., tp);}

        is_key_down(KeyCode::Q)
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Lenia".to_owned(),
        fullscreen: false,
        window_width: MAP_SIZE.0 as i32,
        window_height: MAP_SIZE.1 as i32,
        sample_count: 16,
        ..Default::default()
    }
}
