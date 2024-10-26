use crate::{lenia::Eco, logger::Logger, utils::FrameTimeAnalyzer};
use macroquad::prelude::*;

pub struct UI {
    fta: FrameTimeAnalyzer,
    pub pause: bool,
    //autotune: bool,
    font: Font,
    cursor: Cursor
}

pub struct Cursor {
    field: isize,  // field number
    field_max: usize,
    layer_num: usize,
    pos_y: f32, // physical position
    parameters_lengths: Vec<usize>,
}

// horrible code, idk how it works
impl Cursor {
    pub fn new() -> Self {
        Self { field: 0, layer_num: 0, field_max: 0, pos_y: 0., parameters_lengths: vec![] }
    }
    pub fn update(&mut self, eco: &mut Eco){
        self.parameters_lengths.clear();
        eco.layers.iter().for_each(|l|{
            self.parameters_lengths.push(l.kernel.parameters.len());
            self.parameters_lengths.push(l.growth_map.parameters.len());
        });
        self.field_max = self.parameters_lengths.iter().sum();

        let mut sum = 0;
        let mut parameter_i = 0;
        let mut active = true;
        let mut idx = 0;
        self.pos_y = 7.5;
        self.parameters_lengths.iter().enumerate().for_each(|(i, l)|{
            if active {idx = 0;}
            for _ in 0..*l {
                if self.field > sum && active { idx += 1; self.pos_y += 1.;}
                else { active = false;}
                if self.field == sum {parameter_i = i;}
                sum += 1;
            }
            if active {
                if i&2==0 {self.pos_y += 3.;}
                else {self.pos_y = 7.5;}
            }
            // division because every two elements are from single layer 
        });
        self.layer_num = parameter_i / 2;
        let is_kernel:bool = parameter_i % 2 == 1;


        if is_key_pressed(KeyCode::Down) { self.field += 1; }
        if is_key_pressed(KeyCode::Up) { self.field -= 1; }
        self.field = self.field.clamp(0, self.field_max as isize - 1);
        
        if is_key_pressed(KeyCode::Left) { 
            if is_kernel {eco.layers[self.layer_num].kernel.parameters[idx] -= 0.01;}
            else {eco.layers[self.layer_num].growth_map.parameters[idx] -= 0.002;}
        }
        if is_key_pressed(KeyCode::Right) { 
            if is_kernel {eco.layers[self.layer_num].kernel.parameters[idx] += 0.01;}
            else {eco.layers[self.layer_num].growth_map.parameters[idx] += 0.002;}
        }
    }
}

// selector choses layer growth map parameters to show and then changes 'em in popup
// History managment, with matrix previews
impl UI {
    pub fn new(font: Font) -> Self {

UI { 
            fta: FrameTimeAnalyzer::new(16), 
            pause: false, 
            font,
            cursor: Cursor::new(),
        }
    }

    pub fn execute(&mut self, uid: &String, eco: &mut Eco, logger: &mut Logger) -> bool {
        let mut tp = TextParams { 
            font: Some(&self.font), 
            font_size: 20, 
            font_scale: 1., 
            font_scale_aspect: 1., 
            rotation: 0., 
            color: WHITE 
        };
        self.fta.add_frame_time(get_frame_time()*1000.);
        draw_text_ex(&(self.fta.smooth_frame_time().round().to_string() + " ms"), 450., 30., tp.clone());

        tp.font_size = 20;
        let mut pos_y = 1.;
        draw_text_ex(&("Delta: ".to_owned() + &eco.delta.to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); pos_y+=1.;
        draw_text_ex(&("Fitness: ".to_owned() + &eco.fitness.to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); pos_y+=1.;

        self.cursor.update(eco);
        let layer_num = self.cursor.layer_num;
        draw_circle(64., self.cursor.pos_y as f32 * tp.font_size as f32, 4., GREEN);

        draw_text_ex(&("Layers amount: ".to_owned() + &eco.layers.len().to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("Layer: ".to_owned() + &layer_num.to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("Growth Map: : ") , 24., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- shape - ".to_owned() + &eco.layers[layer_num].growth_map.shape.to_string()) , 
            48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- params - ".to_owned()) , 48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        eco.layers[layer_num].growth_map.parameters.iter().for_each(|p|{
            draw_text_ex(&( ((p * 1000.).round() / 1000.).to_string() ), 72., pos_y * tp.font_size as f32, tp.clone()); 
            pos_y+=1.;
        });
        draw_text_ex(&("Kernel: ") , 24., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- shape - ".to_owned() + &eco.layers[layer_num].kernel.shape.to_string()) , 48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- params - ".to_owned()) , 48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        eco.layers[layer_num].kernel.parameters.iter().for_each(|p|{
            draw_text_ex(&( ((p * 1000.).round() / 1000.).to_string() ), 72., pos_y * tp.font_size as f32, tp.clone()); 
            pos_y+=1.;
        });
        
        draw_text_ex(&("Channel number: ".to_owned() + &eco.layers[layer_num].channel_id.to_string()) , 
            48., pos_y * tp.font_size as f32, tp.clone());

        
        tp.font_size = 64;
        if self.pause {draw_text_ex("PAUSE", 512., 32., tp);}

        self.decorations();
        self.input_handler(uid, eco, logger)
    }

    fn decorations(&mut self) {
        draw_line(512., 0., 512., 1024., 4., BLACK);
    }

    fn input_handler(&mut self, uid: &String, eco: &mut Eco, logger: &mut Logger) -> bool {
        if is_key_pressed(KeyCode::A) {self.pause = !self.pause;}
        
        if is_key_pressed(KeyCode::R) {eco.layers[0].growth_map.parameters[0] += 0.01;}  // width
        if is_key_pressed(KeyCode::E) {eco.layers[0].growth_map.parameters[0] -= 0.01;}
        if is_key_pressed(KeyCode::F) {eco.layers[0].growth_map.parameters[1] += 0.002;}  // offset
        if is_key_pressed(KeyCode::D) {eco.layers[0].growth_map.parameters[1] -= 0.002;}

        if is_key_pressed(KeyCode::S) { logger.save_to_file(); }
        if is_key_pressed(KeyCode::L) { logger.load_from_file(); }
        if is_key_pressed(KeyCode::U) { logger.update_correlation(uid, &eco, (true, true)); }
        


        is_key_down(KeyCode::Q)
    }
}

// id for everyone, needs to be uniqe
