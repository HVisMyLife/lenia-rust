//#![allow(clippy::ptr_arg)]
use macroquad::prelude::*;

mod eco;
use eco::{Ecosystem, Best};

mod utils;
use utils::{FrameTimeAnalyzer, load, save};


// size of map for convolution, it is later padded for warp func, so we have to cut it for now
pub const MAP_SIZE: (i32, i32) = (1280, 820);
pub const SEED: u64 = 1;

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(SEED);
    let mut eco = Ecosystem::new((MAP_SIZE.0 as usize, MAP_SIZE.1 as usize));
    let font = macroquad::text::load_ttf_font("font.ttf").await.unwrap();
    let mut ui = UI::new(font, &mut eco);

    loop {
        clear_background(Color::from_rgba(24, 24, 24, 255));
        let rtx = eco.run(&ui.pause);
        let tx = Texture2D::from_rgba8(*rtx.0, *rtx.1, rtx.2);
        draw_texture(&tx, 0., 0., WHITE);
        if ui.execute(&mut eco) {break};
        next_frame().await
    }
}

struct UI {
    fta: FrameTimeAnalyzer,
    selection: i32,
    pub pause: bool,
    autotune: bool,
    best: Best,
    font: Font,
}

impl UI {
    fn new(font: Font, eco: &mut Ecosystem) -> Self {
        
        UI { 
            fta: FrameTimeAnalyzer::new(16), 
            selection: 0, 
            pause: false, 
            autotune: false,
            best: Best {
                g_params: [eco.layer[0].g_params, eco.layer[1].g_params],
                cycles: 0,
            },
            font,
        }
    }
    fn execute(&mut self, eco: &mut Ecosystem) -> bool {
        if self.autotune && (eco.fitness == 0. || eco.fitness > 5.0) {
            if self.best.compare(eco) { 
                save("data/data", ( 
                    &eco.map, 
                    [
                        (eco.layer[0].size, eco.layer[0].g_params),
                        (eco.layer[1].size, eco.layer[1].g_params)
                    ] 
                ) );
            }
            ( 
                eco.map, 
                [
                    (eco.layer[0].size, eco.layer[0].g_params),
                    (eco.layer[1].size, eco.layer[1].g_params)
                ] 
            ) = load("data/data");
        }
        if is_key_pressed(KeyCode::T) {self.autotune = !self.autotune;}  // enable autotune
        if is_key_pressed(KeyCode::G) {self.best.cycles = 0;} // reset best score, current better
        if is_key_pressed(KeyCode::F) {eco.cycles = 0; eco.fitness = 50.} // reset current score
        if is_key_pressed(KeyCode::A) {self.pause = !self.pause;}
        if is_key_pressed(KeyCode::S) {
            save("data/data", ( 
                    &eco.map, 
                    [
                        (eco.layer[0].size, eco.layer[0].g_params),
                        (eco.layer[1].size, eco.layer[1].g_params)
                    ] 
                ) );
        } // save current gen
        if is_key_pressed(KeyCode::L) { // load saved gen
            ( 
                eco.map, 
                [
                    (eco.layer[0].size, eco.layer[0].g_params),
                    (eco.layer[1].size, eco.layer[1].g_params)
                ] 
            ) = load("data/data");
        }
        // paramenters menu interactions
        if is_key_pressed(KeyCode::Up) {self.selection-=1;}
        if is_key_pressed(KeyCode::Down) {self.selection+=1;}
        self.selection = self.selection.clone().clamp(0, 5);
        if self.autotune && (eco.fitness == 0. || eco.fitness > 5.0) {self.best.compare(eco);}
        let mut change = 0.;

        if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::Left) {
            if is_key_pressed(KeyCode::Left) {change=-0.001;}
            if is_key_pressed(KeyCode::Right) {change=0.001;}
            match self.selection {
                0 => eco.layer[0].g_params.0 = ((eco.layer[0].g_params.0 + change) * 1000.).round() as f32 / 1000.,
                1 => eco.layer[0].g_params.1 = ((eco.layer[0].g_params.1 + change) * 1000.).round() as f32 / 1000.,
                2 => eco.layer[0].g_params.2 = ((eco.layer[0].g_params.2 + change) * 1000.).round() as f32 / 1000.,
                3 => eco.layer[1].g_params.0 = ((eco.layer[1].g_params.0 + change) * 1000.).round() as f32 / 1000.,
                4 => eco.layer[1].g_params.1 = ((eco.layer[1].g_params.1 + change) * 1000.).round() as f32 / 1000.,
                5 => eco.layer[1].g_params.2 = ((eco.layer[1].g_params.2 + change) * 1000.).round() as f32 / 1000.,
                _ => {},
            }
        }
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
        
        draw_text_ex(&("deltaT0: ".to_owned() + &eco.layer[0].g_params.0.to_string()), 8., 30.*2., tp.clone());
        draw_text_ex(&("offset0: ".to_owned() + &eco.layer[0].g_params.1.to_string()), 8., 30.*3., tp.clone());
        draw_text_ex(&("width0:  ".to_owned() + &eco.layer[0].g_params.2.to_string()), 8., 30.*4., tp.clone());
        draw_text_ex(&("deltaT1: ".to_owned() + &eco.layer[1].g_params.0.to_string()), 8., 30.*5., tp.clone());
        draw_text_ex(&("offset1: ".to_owned() + &eco.layer[1].g_params.1.to_string()), 8., 30.*6., tp.clone());
        draw_text_ex(&("width1:  ".to_owned() + &eco.layer[1].g_params.2.to_string()), 8., 30.*7., tp.clone());
        draw_circle(0., 50. + (30 * self.selection) as f32, 10., RED);

        draw_text_ex(&("%: ".to_owned() + &eco.fitness.to_string()), 8., 30.*9., tp.clone());
        draw_text_ex(&("t: ".to_owned() + &eco.cycles.to_string()), 8., 30.*10., tp.clone());
        draw_text_ex(&("autotune: ".to_owned() + &self.autotune.to_string()), 8., 30.*11., tp.clone());
        tp.font_size = 128;
        if self.pause {draw_text_ex("PAUSE", 256., 256., tp);}
        
        is_key_down(KeyCode::Q)
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Lenia".to_owned(),
        fullscreen: false,
        window_width: MAP_SIZE.0,
        window_height: MAP_SIZE.1,
        sample_count: 16,
        ..Default::default()
    }
}

// Legacy code

        // padding on borders, so convolution will be continous
        // self.map_save = pad(&self.map, &[[K_MAX as usize; 2]], PadMode::Wrap);
        // 
        //
        // // convolving using fft and cutting borders
        // self.nh_sum = fftconvolve(&self.map_save, &self.kernel, Mode::Same)
        //     .unwrap()
        //     .slice(s![K_MAX..MAP_SIZE.0 + K_MAX, K_MAX..MAP_SIZE.1 + K_MAX])
        //     .to_owned();


        
        //for ny in 0..=K_R as usize*2 {
        //    for nx in 0..=K_R as usize*2 {
        //        if self.kernel[[nx, ny]] != 0.0 {
        //            graphics.draw_rectangle(
        //                Rectangle::new(
        //                    Vector2::new(nx as f32*5.0, ny as f32*5.0), 
        //                    Vector2::new(nx as f32*5.0+5.0, ny as f32*5.0+5.0)), 
        //                Color::from_rgb(self.kernel[[nx, ny]]*1.0, 0.0, 0.0)
        //            );
        //        } else {
        //            graphics.draw_rectangle(
        //                Rectangle::new(
        //                    Vector2::new(nx as f32*5.0, ny as f32*5.0), 
        //                    Vector2::new(nx as f32*5.0+5.0, ny as f32*5.0+5.0)), 
        //                Color::from_rgb(0.0, 255.0, 0.0)
        //            );
        //        }
        //    }
        //}

// lenia alg
//pub fn fng(map: &Vec<Vec<f32>>, k_values: &[f32], g_params: &(f32,f32,f32), id: &(i32, i32)) -> f32 {
//    let mut count = 0.0;
//    let mut n = 0;
//    for ny in -K_R..=K_R {
//        //let d = f32::sqrt((K_R * K_R - ny * ny) as f32) as i32;
//        //for nx in -d..=d {
//        for nx in -K_R..=K_R {
//            count +=
//            map[(id.0 + nx + K_MAX) as usize][(id.1 + ny + K_MAX) as usize]
//            *k_values[n];// kernel shape
//            n+=1;
//        }
//    }
//    
//    (
//        map[(id.0 + K_MAX) as usize][(id.1 + K_MAX) as usize] 
//        + g_params.0 * (E.powf( -((count-g_params.1).powi(2)) / g_params.2.powi(2) ) * 2.0 -1.0)
//    ).clamp(0.0, 1.0)     // clamping between 0-1: A + dtG(A*K)
//
//}

        // looping map in x and y, so entities can "tunnel" itself through borders
        //self.map_save = self.map.clone();

        //self.map_save.par_iter_mut().enumerate().for_each(|(i, e)| {
        //    e.splice(0..0, self.map[i].split_at(self.map[i].len()-K_MAX as usize).1.iter().cloned());
        //    e.extend_from_slice(self.map[i].split_at(K_MAX as usize).0);
        //});
        //let map_save2 = self.map_save.to_vec();
        //self.map_save.splice(0..0, map_save2.split_at(map_save2.len()-K_MAX as usize).1.iter().cloned());
        //self.map_save.extend_from_slice(map_save2.split_at(K_MAX as usize).0); // extending
        
        //// counting active entities
        //let mut act: Vec<(i32, i32)> = vec![];
        //self.map_save.iter_mut().enumerate().for_each(|(i, x)| {
        //    x.iter_mut().enumerate().for_each(|(j, y)|{
        //       if *y > 0.0 { act.push((i as i32-K_MAX, j as i32-K_MAX)); }
        //    });
        //});
 
        //// choose between optimized (<2% act) and normal mode
        //if act.len() < (MAP_SIZE.0 * MAP_SIZE.1) as usize / TURBO_TRESHOLD {
        //    // converting sequential iterator to parellel using rayon (two times)
        //    self.map.par_iter_mut().enumerate().for_each(|(i, x)| {
        //        let p1 = i as i32 - K_MAX;
        //        let p2 = i as i32 + K_MAX;
        //        x.par_iter_mut().enumerate().for_each(|(j, y)|{
        //        let p3 = j as i32 - K_MAX;
        //        let p4 = j as i32 + K_MAX;
        //            for h in act.iter() {
        //                if h.0 > p1 && h.0 < p2 && h.1 > p3 && h.1 < p4 {
        //                    *y = fng(&self.map_save, &self.k_values, &self.g_params, &(i as i32, j as i32));
        //                    break;
        //                }
        //            }
        //        });
        //    });
        //} else {
        //    // converting sequential iterator to parellel using rayon (two times)
        //    self.map.par_iter_mut().enumerate().for_each(|(i, x)| {
        //        x.par_iter_mut().enumerate().for_each(|(j, y)|{
        //            *y = fng(&self.map_save, &self.k_values, &self.g_params, &(i as i32, j as i32));
        //        });
        //    });
        //} 
        //

//pub fn scale_img(map: &[u8]) -> Vec<u8> {
//    let mut img = vec![];
//    let mut img2 = vec![];
//  
//    for i in 0..map.len()/4 {
//        img.push(map[i*4]);
//        img.push(map[i*4+1]);
//        img.push(map[i*4+2]);
//        img.push(map[i*4+3]);
//        img.push(map[i*4]);
//        img.push(map[i*4+1]);
//        img.push(map[i*4+2]);
//        img.push(map[i*4+3]);
//    }
//    for i in 0..(img.len()) / (MAP_SIZE.0 * 2 * 4) as usize {
//        let h = img.split_at(i*(MAP_SIZE.0 * 2 * 4) as usize).1.split_at((MAP_SIZE.0 * 2 * 4) as usize).0;
//        img2.extend_from_slice(h);
//        img2.extend_from_slice(h);
//    }
//  
//    img2
//} 


//fn check_state(map: &Vec<Vec<f32>>) -> bool {
//    let  mut count: f32 = 0.0;
//    (0..map.len()).for_each(|x| {
//        (0..map[x].len()).for_each(|y| {
//            count += map[x][y];
//        });
//    });
//    println!("Count: {}", count);
//    !(1.0..=15000.0).contains(&count)   
//}

//fn print_params(g_params: &(f32,f32,f32)){
//    println!("||");
//    println!("{} \t- Kernel radius", K_R);
//    println!("{} \t- Parameter dt", g_params.0);
//    println!("{} \t- G func center", g_params.1);
//    println!("{} \t- G func width", g_params.2);
//}
//
//if is_key_down(KeyCode::Q) {process::exit(1);} // end simulation when "q" pressed
        //if is_key_down(KeyCode::Space) {
        //    if is_key_down(KeyCode::Up) {
        //        if is_key_down(KeyCode::A) {g_params.0 += 0.001; println!("{}\t - dt", g_params.0);}
        //        if is_key_down(KeyCode::S) {g_params.1 += 0.05; println!("{}\t - g_center", g_params.1);}
        //        if is_key_down(KeyCode::D) {g_params.2 += 0.05; println!("{}\t - g_width", g_params.2);}
        //    }   
        //    else if is_key_down(KeyCode::Down) {   
        //        if is_key_down(KeyCode::A) {g_params.0 -= 0.001; println!("{}\t - dt", g_params.0);}
        //        if is_key_down(KeyCode::S) {g_params.1 -= 0.05; println!("{}\t - g_center", g_params.1);}
        //        if is_key_down(KeyCode::D) {g_params.2 -= 0.05; println!("{}\t - g_width", g_params.2);}
        //    }
        //}
