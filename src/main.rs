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
use lenia::{Channel, Eco, Function, Layer, Shape, Destiny};

// TODO: UI with fitness graph, tuning AI

fn creator(size: (usize, usize)) -> Eco {
    let mut eco = Eco::new((size.0, size.1), 0.1, 0, vec![], vec![]);
    let mut matrix = Array2::<f32>::zeros(size);

    // generate random starting point
    for x in (size.0 as f32 * 0.3)as usize..(size.0 as f32 * 0.6)as usize{ 
        for y in (size.1 as f32 * 0.3)as usize..(size.1 as f32 * 0.6)as usize{ 
            matrix[[x as usize, y as usize]] = rand::gen_range(0.0,1.0);
        }
    } 
    eco.channels.push( Channel::new(matrix) );

    eco.layers.push(
        Layer::new(
            Function::new(64, Shape::GaussianBump, false, Some(vec![0.5, 0.15]), Some(Destiny::Kernel)), 
            Function::new(64, Shape::GaussianBump, true, Some(vec![0.15, 0.02]), Some(Destiny::GrowthMap)), 
            0
        ) 
    );
    eco.layers.push(
        Layer::new(
            Function::new(64, Shape::TripleBump, false, None, Some(Destiny::Kernel)), 
            Function::new(64, Shape::GaussianBump, true, Some(vec![0.15, 0.02]), Some(Destiny::GrowthMap)), 
            0
        ) 
    );
    eco
}

#[macroquad::main(window_conf)]
async fn main() {
    let window_size: (usize, usize) = (1024- (64*2), 1024- (64*2) );
    let font = macroquad::text::load_ttf_font("font.ttf").await.unwrap();

    let mut eco = creator(window_size);
    eco.init();

    // LOGGER
    let mut logger = Logger::new();
    logger.load_from_file();
    logger.push_instance(eco.clone());
    logger.update_instance(0, eco.clone());
    logger.pop_instance(0);
    logger.push_instance(eco.clone());
    eco = logger.get_instance(0).unwrap();
    eco.init();
    logger.save_to_file();
    // LOGGER

    let mut ui = UI::new(font);

    request_new_screen_size(window_size.0 as f32, window_size.1 as f32);

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
struct Logger {
    instances: Vec<(TomlData, Vec<Array2<f32>>)> // parameters and matrixes
}

impl Logger {
    fn new() -> Self {
        Self { instances: vec![] }
    }

    // returns pushed instance index
    fn push_instance(&mut self, eco: Eco) -> usize {
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
    fn get_instance(&self, idx: usize) -> Option<Eco> {
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
    fn update_instance(&mut self, idx: usize, eco: Eco) -> bool {
        if idx >= self.instances.len() {return false;}
        self.push_instance(eco);
        self.instances[idx] = self.instances.last().unwrap().clone();
        self.instances.pop();
        true
    }

    // Return false if there is no instance at index
    fn pop_instance(&mut self, idx: usize) -> bool {
        if idx >= self.instances.len() {return false;}
        self.instances.remove(idx);
        true
    }

    // overwrites old saves, so it's recommended to load before saving, returns amount of records
    fn save_to_file(&self) -> usize {
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
    fn load_from_file(&mut self) -> usize {
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
        window_width: 1024,
        window_height: 1024,
        sample_count: 16,
        ..Default::default()
    }
}
