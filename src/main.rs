//#![allow(clippy::ptr_arg)]
use macroquad::prelude::*;
use ndarray::prelude::*;

mod utils;

mod lenia;
use lenia::{Channel, Eco, Function, Layer, Shape};

mod logger;
use logger::Logger;

mod ui;
use ui::UI;

// TODO: UI creator with fitness graph, tuning AI

fn _creator(size: (usize, usize)) -> Eco {
    let mut eco = Eco::new((size.0, size.1), 0.1, 0, vec![], vec![]);
    let mut matrix = Array2::<f32>::zeros(size);

    // generate random starting point
    for x in (size.0 as f32 * 0.1)as usize..(size.0 as f32 * 0.2)as usize{ 
        for y in (size.1 as f32 * 0.1)as usize..(size.1 as f32 * 0.2)as usize{ 
            matrix[[x as usize, y as usize]] = rand::gen_range(0.0,1.0);
        }
    } 
    for x in (size.0 as f32 * 0.7)as usize..(size.0 as f32 * 0.8)as usize{ 
        for y in (size.1 as f32 * 0.4)as usize..(size.1 as f32 * 0.55)as usize{ 
            matrix[[x as usize, y as usize]] = rand::gen_range(0.0,1.0);
        }
    } 
    eco.channels.push( Channel::new(matrix.clone()) );

    // width, offset
    //let mut f = vec![];
    //f.push( Function::new(64, Shape::GaussianBump, false, vec![0.12, 0.5], Some(Destiny::Kernel)) );
    //f.push( Function::new(64, Shape::ExponentialDecay, false, vec![0.18, 0.0], Some(Destiny::Kernel)) );
    //f.push( Function::new(64, Shape::SmoothTransition, false, vec![0.10, 0.5], Some(Destiny::Kernel)) );
    //f.push( Function::new(64, Shape::MexicanHat, false, vec![0.16, 0.5], Some(Destiny::Kernel)) );
    //f.push( Function::new(64, Shape::TripleBump, false, vec![0.06, 0.5], Some(Destiny::Kernel)) );

    eco.layers.push(
        Layer::new(
            Function::new(Shape::MexicanHat, false, vec![0.16, 0.5]), 
            Function::new(Shape::GaussianBump, true, vec![0.12, 0.5]), 
            0, 64
        ) 
    );
    eco
}

#[macroquad::main(window_conf)]
async fn main() {
    let window_size: (usize, usize) = (1024- (64*2), 1024- (64*2) );
    let ui_offset = 512.;
    let font = macroquad::text::load_ttf_font("font.ttf").await.unwrap();

    // LOGGER
    let mut logger = Logger::new();
    logger.load_from_file();
    let mut uid = "sr6X529DRyGIS1bqOSydCR".to_string();
    let mut eco = _creator(window_size);//logger.get_correlation(&uid).unwrap();
    eco.init();
    //logger.push_correlation(&eco, "MK".to_string());
    //logger.save_to_file();
    // LOGGER

    let mut ui = UI::new(font);

    request_new_screen_size(window_size.0 as f32 + ui_offset, window_size.1 as f32);

    loop {
        clear_background(Color::from_rgba(24, 24, 24, 255));
        if !ui.pause {
            eco.evaluate();
        }

        let rtx = logger.image(&eco.channels[0].matrix);       
        let tx = Texture2D::from_rgba8(rtx.0.0 as u16, rtx.0.1 as u16, rtx.1);
        draw_texture(&tx, ui_offset, 0., WHITE);
        //draw_texture_ex(&tx, ui_offset, 0., WHITE, DrawTextureParams{
        //    dest_size: Some(Vec2::new(1080., 1080.)),
        //    ..Default::default()
        //});

        if ui.execute(&mut uid, &mut eco, &mut logger) {break};
        next_frame().await
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
