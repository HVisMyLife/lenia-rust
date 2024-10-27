use std::{fs::File, io::prelude::*, time::{Duration, SystemTime}};

use crate::{lenia::Eco, logger::Logger, utils::FrameTimeAnalyzer};
use macroquad::prelude::*;


pub struct Menu {
    pub active: bool,
    field: isize,
}

impl Menu {
    pub fn new() -> Self {
        Self { active: true, field:0 }
    }

    pub fn run(&mut self, uid: &mut String, eco: &mut Eco, logger: &mut Logger, tp: TextParams){
        let tp = tp;
        let list = logger.get_correlation_list();
        if is_key_pressed(KeyCode::Down) { self.field += 1; }
        if is_key_pressed(KeyCode::Up) { self.field -= 1; }
        self.field = self.field.clamp(0, list.len() as isize - 1);

        let mut pos_y = 3.;
        draw_text_ex(&"Menu: ".to_string(), 8., tp.font_size as f32 * pos_y, tp.clone());
        pos_y += 1.;
        list.iter().enumerate().for_each(|(i, c)|{
            draw_text_ex(&(i.to_string() + ") " + &c.0 + "_" + &c.1), 24., tp.font_size as f32 * pos_y, tp.clone());
            pos_y += 1.;
        });
        draw_circle(30., tp.font_size as f32 * (self.field as f32 + 3.7), 8., GREEN);
        draw_rectangle(512., 0., 1024., 1024., Color::from_rgba(24, 24, 24, 255));
        let path: String = ("data/matrix/".to_string() + &list[self.field as usize].2 + ".png").to_owned();
        let mut file = File::open(path).unwrap();
        let mut bytes = vec![];
        file.read_to_end(&mut bytes).unwrap();
        let img = Image::from_file_with_format(
            &bytes,
            Some(ImageFormat::Png)
            ).unwrap();
        let tx = Texture2D::from_image( &img );
        draw_texture(&tx, 512., 0., WHITE);

        if is_key_pressed(KeyCode::Enter) {
            uid.clear();
            uid.push_str(&list[self.field as usize].0);
            *eco = logger.get_correlation(&list[self.field as usize].0).unwrap();
            eco.init();
            self.active = false;
        }
        if is_key_pressed(KeyCode::Delete) {
            logger.pop_correlation(&list[self.field as usize].0);
            logger.save_to_file();
            logger.load_from_file();
        }
        if is_key_pressed(KeyCode::Insert) {
            *eco = logger.get_correlation(&list[self.field as usize].0).unwrap();
            uid.clear();
            uid.push_str(logger.push_correlation(eco, "new".to_string()));

            logger.save_to_file();
            logger.load_from_file();
        }

    }
}

pub struct UI {
    fta: FrameTimeAnalyzer,
    pub pause: bool,
    popup: Popup,
    font: Font,
    dd: DynamicDisplay,
    menu: Menu
}

// selector choses layer growth map parameters to show and then changes 'em in popup
// History managment, with matrix previews
impl UI {
    pub fn new(font: Font) -> Self {
        UI { 
            fta: FrameTimeAnalyzer::new(16), 
            pause: false, 
            popup: Popup::new(),
            font,
            dd: DynamicDisplay::new(),
            menu: Menu::new(),
        }
    }

    pub fn execute(&mut self, uid: &mut String, eco: &mut Eco, logger: &mut Logger) -> bool {

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

        if !self.menu.active {self.dd.update(uid, eco, tp.clone());}
        else {self.menu.run(uid, eco, logger, tp.clone());}

        self.popup.update(tp.clone());
        if self.menu.active {self.pause = true;}  // pause simulation when menu
        tp.font_size = 64;
        if self.pause {draw_text_ex("PAUSE", 24., 850., tp.clone());}

        self.decorations();
        self.input_handler(uid, eco, logger)
    }

    fn decorations(&mut self) {
        draw_line(512., 0., 512., 1024.-128., 4., BLACK);
    }

    fn input_handler(&mut self, uid: &mut String, eco: &mut Eco, logger: &mut Logger) -> bool {
        if is_key_pressed(KeyCode::A) {self.pause = !self.pause;}
        
        if is_key_pressed(KeyCode::R) {eco.layers[0].growth_map.parameters[0] += 0.01;}  // width
        if is_key_pressed(KeyCode::E) {eco.layers[0].growth_map.parameters[0] -= 0.01;}
        if is_key_pressed(KeyCode::F) {eco.layers[0].growth_map.parameters[1] += 0.002;}  // offset
        if is_key_pressed(KeyCode::D) {eco.layers[0].growth_map.parameters[1] -= 0.002;}

        if is_key_pressed(KeyCode::S) { logger.save_to_file(); self.popup.show(&"SAVED".to_string(), None);}
        if is_key_pressed(KeyCode::L) { logger.load_from_file(); self.popup.show(&"LOADED".to_string(), None);}
        if is_key_pressed(KeyCode::U) { 
            logger.update_correlation(uid, &eco, (true, true)); 
            self.popup.show(&"UPDATED".to_string(), None);
        }
        if is_key_pressed(KeyCode::N) { 
            uid.clear();
            uid.push_str(logger.push_correlation(&eco, "new".to_string())); 
            self.popup.show(&"SAVED to NEW".to_string(), None);
        }
        if is_key_pressed(KeyCode::Escape) { 
            logger.save_to_file();
            logger.load_from_file();
            self.menu.active = true;
        }
        

        is_key_down(KeyCode::Q)
    }

}

// Shows text for set duration
pub struct Popup {
    text: String,
    duration: Duration,
    show_time: SystemTime
}

impl Popup {
    pub fn new() -> Self {
        Self { text: String::new(), duration: Duration::new(0,0), show_time: SystemTime::now() }
    }
    
    pub fn show(&mut self, text: &String, duration: Option<Duration>) {
        self.show_time = SystemTime::now();
        self.duration = match duration {
            Some(d) => d,
            None => Duration::new(2, 0),
        };
        self.text.clear();
        self.text.push_str(text);
    }

    pub fn update(&self, tp: TextParams) {
        if self.show_time.elapsed().unwrap() < self.duration {  // if duration elapsed don't show
            let mut tp = tp;
            tp.font_size = 48;
            draw_line(
                128., 768. + 5., 
                128. + self.text.chars().count() as f32 * tp.font_size as f32 * 0.5, 768. + 5., 
                4., WHITE
            );
            draw_text_ex(&self.text, 128., 768., tp);
        } 
    }
}


pub struct DynamicDisplay {
    field: isize,  // field number
    field_old: isize,
    field_max: usize,
    layer_num: usize,
    is_kernel: bool,
    idx: usize,
    uid_old: String, // for checking if there is need to recalculate
    pos_y: f32, // physical position
    parameters_lengths: Vec<usize>,
    kernel_shape: [f32; 100],
    growth_map_shape: [f32; 100],
}

// horrible code, idk how it works
impl DynamicDisplay {
    pub fn new() -> Self {
        Self { field: 0, field_old: -1, layer_num: 0, is_kernel: false, idx: 0, uid_old: String::new(), field_max: 0, pos_y: 0., 
            parameters_lengths: vec![], 
            kernel_shape: [0.;100], growth_map_shape: [0.;100] 
        }
    }
    pub fn update(&mut self, uid: &String, eco: &mut Eco, tp: TextParams){
        let mut tp = tp;

        if self.field != self.field_old || self.uid_old != *uid {
            self.parameters_lengths.clear();
            eco.layers.iter().for_each(|l|{
                self.parameters_lengths.push(l.kernel.parameters.len());
                self.parameters_lengths.push(l.growth_map.parameters.len());
            });
            self.field_max = self.parameters_lengths.iter().sum();

            let mut sum = 0;
            let mut parameter_i = 0;
            let mut active = true;
            self.idx = 0;
            self.pos_y = 8.5;
            self.parameters_lengths.iter().enumerate().for_each(|(i, l)|{
                if active {self.idx = 0;}
                for _ in 0..*l {
                    if self.field > sum && active { self.idx += 1; self.pos_y += 1.;}
                    else { active = false;}
                    if self.field == sum {parameter_i = i;}
                    sum += 1;
                }
                if active {
                    if i&2==0 {self.pos_y += 3.;}
                    else {self.pos_y = 8.5;}
                }
                // division because every two elements are from single layer 
            });
            self.layer_num = parameter_i / 2;
            self.is_kernel = parameter_i % 2 == 1;
            
            // after possible layer change and after parameter change (below)
            self.kernel_shape.iter_mut().enumerate().for_each(|(i, x)| *x = 100. * eco.layers[self.layer_num].kernel.calc(i as f32/100.) );
            self.growth_map_shape.iter_mut().enumerate().for_each(|(i, x)| *x = 100. * eco.layers[self.layer_num].growth_map.calc(i as f32/100.) );
        }
        self.field_old = self.field;
        self.uid_old = uid.to_string();


        if is_key_pressed(KeyCode::Down) { self.field += 1; }
        if is_key_pressed(KeyCode::Up) { self.field -= 1; }
        self.field = self.field.clamp(0, self.field_max as isize - 1);
        if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::Right){
            let mut value = 0.02;
            if self.idx == 0 {value = 0.002;}
            if is_key_pressed(KeyCode::Left) {value = -value;}
            if self.is_kernel {
                eco.layers[self.layer_num].kernel.parameters[self.idx] += value;
                eco.init();  // need to regenerate kernel lookup
            }
            else {eco.layers[self.layer_num].growth_map.parameters[self.idx] += value;}
            self.kernel_shape.iter_mut().enumerate().for_each(|(i, x)| *x = 100. * eco.layers[self.layer_num].kernel.calc(i as f32/100.) );
            self.growth_map_shape.iter_mut().enumerate().for_each(|(i, x)| *x = 100. * eco.layers[self.layer_num].growth_map.calc(i as f32/100.) );
        } 



        tp.font_size = 20;
        let mut pos_y = 1.;
        draw_text_ex(&("Correlation: ".to_owned() + uid) , 8., pos_y * tp.font_size as f32, tp.clone()); pos_y+=1.;
        draw_text_ex(&("Delta: ".to_owned() + &eco.delta.to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); pos_y+=1.;
        draw_text_ex(&("Fitness: ".to_owned() + &eco.fitness.to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); pos_y+=1.;

        let layer_num = self.layer_num;
        draw_circle(64., self.pos_y as f32 * tp.font_size as f32, 4., GREEN);

        draw_text_ex(&("Layers amount: ".to_owned() + &eco.layers.len().to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("Layer: ".to_owned() + &layer_num.to_string()) , 8., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("Growth Map: : ") , 24., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- shape - ".to_owned() + &eco.layers[layer_num].growth_map.shape.to_string()) , 
            48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- params (width, offset) - ".to_owned()) , 48., pos_y * tp.font_size as f32, tp.clone()); 

        self.growth_map_shape.iter().enumerate().for_each(|(x,y)|{
            draw_rectangle(300.+x as f32 * 2., pos_y * tp.font_size as f32, 2., -*y, BLACK);
        });
        draw_rectangle(300., pos_y * tp.font_size as f32 - 0.5, 100. * 2., 1., Color::from_rgba(255, 255, 255, 48));
        draw_rectangle(300. + 100. - 0.5, pos_y * tp.font_size as f32 - 100., 1., 200., Color::from_rgba(255, 255, 255, 48));

        pos_y+=1.;
        eco.layers[layer_num].growth_map.parameters.iter().for_each(|p|{
            draw_text_ex(&( ((p * 1000.).round() / 1000.).to_string() ), 72., pos_y * tp.font_size as f32, tp.clone()); 
            pos_y+=1.;
        });
        draw_text_ex(&("Kernel: ") , 24., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- shape - ".to_owned() + &eco.layers[layer_num].kernel.shape.to_string()) , 48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        draw_text_ex(&("- params (width, offset) - ".to_owned()) , 48., pos_y * tp.font_size as f32, tp.clone()); 
        pos_y+=1.;
        eco.layers[layer_num].kernel.parameters.iter().for_each(|p|{
            draw_text_ex(&( ((p * 1000.).round() / 1000.).to_string() ), 72., pos_y * tp.font_size as f32, tp.clone()); 
            pos_y+=1.;
        });
        draw_text_ex(&("Channel number: ".to_owned() + &eco.layers[layer_num].channel_id.to_string()) , 
            24., pos_y * tp.font_size as f32, tp.clone());
        
        pos_y += 3.;
        self.kernel_shape.iter().enumerate().for_each(|(x,y)|{
            draw_rectangle(300.+x as f32 * 2., pos_y * tp.font_size as f32, 2., -*y, BLACK);
        });
        draw_rectangle(300., pos_y * tp.font_size as f32 - 0.5, 100. * 2., 1., Color::from_rgba(255, 255, 255, 48));
        draw_rectangle(300. + 100. - 0.5, pos_y * tp.font_size as f32 - 100., 1., 200., Color::from_rgba(255, 255, 255, 48));

    }
}
