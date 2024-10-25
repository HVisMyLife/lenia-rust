use crate::{lenia::Eco, logger::Logger, utils::FrameTimeAnalyzer};
use macroquad::prelude::*;

pub struct UI {
    fta: FrameTimeAnalyzer,
    pub pause: bool,
    //autotune: bool,
    font: Font,
}

impl UI {
    pub fn new(font: Font) -> Self {

        UI { 
            fta: FrameTimeAnalyzer::new(16), 
            pause: false, 
            font,
        }
    }

    pub fn execute(&mut self, uid: &String, eco: &mut Eco, logger: &mut Logger) -> bool {
        if is_key_pressed(KeyCode::A) {self.pause = !self.pause;}
        
        if is_key_pressed(KeyCode::R) {eco.layers[0].growth_map.parameters[0] += 0.01;}  // width
        if is_key_pressed(KeyCode::E) {eco.layers[0].growth_map.parameters[0] -= 0.01;}
        if is_key_pressed(KeyCode::F) {eco.layers[0].growth_map.parameters[1] += 0.002;}  // offset
        if is_key_pressed(KeyCode::D) {eco.layers[0].growth_map.parameters[1] -= 0.002;}

        if is_key_pressed(KeyCode::S) { logger.save_to_file(); }
        if is_key_pressed(KeyCode::L) { logger.load_from_file(); }
        if is_key_pressed(KeyCode::U) { logger.update_correlation(uid, &eco, (true, true)); }
        
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

// id for everyone, needs to be uniqe
