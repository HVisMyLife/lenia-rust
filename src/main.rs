#![allow(clippy::ptr_arg)]
use std::f32::consts::E;
use std::time;
use rayon::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use speedy2d::color::Color;
use speedy2d::dimen::Vector2;
use speedy2d::shape::Rectangle;
use speedy2d::window::{
    WindowHandler,
    WindowHelper,
    WindowStartupInfo
};
use speedy2d::{Graphics2D, Window};
use speedy2d::font::{Font, TextOptions};
use speedy2d::font::TextLayout;
use ndarray::{prelude::*, Zip};
use fftconvolve::{fftconvolve, Mode};


pub const DISPLAY_RES: (u32, u32) = (1000, 1000); // should be a multiplication of MAP_SIZE

pub const MAP_SIZE: (i32, i32) = (1000, 1000);
pub const SEED: u64 = 2;

pub const K_MAX: i32 = 20;             // maximum radius of kernel any
pub const K_R: i32 = 40;               // radius of kernel no.0

pub const DT: f32 = 10.0;      // % add each cycle
pub const CENTER: f32 = 24.0;  // % of neighbours full
pub const WIDTH: f32 = 3.46;   // width



// calc for kernel matrix values
pub fn kernel_calc(radius: i32) -> Array2<f32> {
    let mut kernel: Array2<f32> = Array2::<f32>::zeros(( (2*radius+1) as usize, (2*radius+1) as usize ));

    for ny in -radius..=radius {
        let d = f32::sqrt((radius * radius - ny * ny) as f32) as i32;
        //for nx in -d..=d {
        for nx in -radius..=radius {
            let r = f32::sqrt((nx.pow(2) + ny.pow(2)) as f32);

            if ny == 0 && nx == 0 || nx < -d || nx > d { kernel[[(nx+radius)as usize, (ny+radius)as usize]] = 0.0;}
            else {kernel[[(nx+radius)as usize, (ny+radius)as usize]] = E.powf( -(r - (radius as f32*1.0)/2.0).powi(2) / (radius as f32 * 1.0) );}// kernel shape
        }
    }

    kernel /= kernel.sum()/10000.0;
    kernel
}


struct MyWindowHandler {
    // main lenia parameters
    g_params: (f32,f32,f32), // dt, g-center, g-width ( 0-100 neighbourhood )
    // precalculated kernel values
    kernel: Array2<f32>,
    
    // texture for drawing states
    texture_slice: Vec<u8>,
    texture_size: Vector2<u32>,
    
    // main array of states
    map: Array2<f32>,     
    //map_save: Array2<f32>,
    map_rect: Rectangle,
    nh_sum: Array2<f32>,
    
    // random generator
    rng: ChaCha8Rng,
    delta: time::Instant,

    // font ??? xd wtf
    font: Font,
    
}

impl MyWindowHandler {
    fn new(seed: u64) -> Self {

        
        
        let bytes = include_bytes!("../font.ttf");
        let font = Font::new(bytes).unwrap();

        Self { 
            g_params: (DT/100.0, CENTER * 100.0, WIDTH*WIDTH*10000.0), // 0-100 neighbourhood
            kernel: kernel_calc(K_R),

            texture_slice: [0;  (4 * MAP_SIZE.1 * MAP_SIZE.0) as usize].to_vec(),
            texture_size: Vector2::new(MAP_SIZE.0 as u32, MAP_SIZE.1 as u32),

            map: Array2::<f32>::zeros((MAP_SIZE.0 as usize, MAP_SIZE.1 as usize)), 
            //map_save: Array2::<f32>::zeros((MAP_SIZE.0 as usize, MAP_SIZE.1 as usize)),
            map_rect: Rectangle::new(Vector2::new(0.0, 0.0), Vector2::new(DISPLAY_RES.0 as f32, DISPLAY_RES.1 as f32)),
            nh_sum: Array2::<f32>::zeros([0, 0]),

            rng: ChaCha8Rng::seed_from_u64(seed),
            delta: time::Instant::now(),
            
            font,
        }
    }
}

impl WindowHandler for MyWindowHandler {
    
    fn on_start(&mut self, helper: &mut WindowHelper, _info: WindowStartupInfo){
        helper.set_cursor_visible(true);
        helper.set_resizable(false);

        for y in self.rng.gen_range((MAP_SIZE.1 as f32/2.7) as i32..MAP_SIZE.1/2)..self.rng.gen_range(MAP_SIZE.1/2..(MAP_SIZE.1 as f32/1.8) as i32) {
            for x in self.rng.gen_range((MAP_SIZE.0 as f32/8.0) as i32..MAP_SIZE.0/2)..self.rng.gen_range(MAP_SIZE.0/2..(MAP_SIZE.0 as f32/1.1) as i32) {
                self.map[[x as usize, y as usize]] = self.rng.gen_range(0.0..1.0);
            }
        } 
    }

    fn on_draw(&mut self, helper: &mut WindowHelper, graphics: &mut Graphics2D){
        self.delta = time::Instant::now();
        graphics.clear_screen(Color::from_rgb(0.2, 0.2, 0.2));

        self.nh_sum = fftconvolve(&self.map, &self.kernel, Mode::Same).unwrap();


        Zip::from(&mut self.map)
            .and(&self.nh_sum)
            .par_for_each(|m, &o| {
                *m = (
                    *m 
                    + self.g_params.0 * (( -((o-self.g_params.1).powi(2)) / self.g_params.2 ).exp() * 2.0 -1.0)
                ).clamp(0.0, 1.0)     // clamping between 0-1: A + dtG(A*K)
            });


        // printing alive entities
        self.texture_slice.par_iter_mut().enumerate().for_each(|(i, x)| {
            let col = (self.map[[(i/4)%MAP_SIZE.0 as usize, (i/4)/MAP_SIZE.0 as usize]] * 255.0) as u8;
            if i%4 == 0{*x = col;}
            //if i%4 == 1{*x = 0;} 
            else if i%4 == 2{*x = 255 - col;}
            else if i%4 == 3{*x = (col*2).clamp(0, 255);}
        });
        
        let img = graphics.create_image_from_raw_pixels(
            speedy2d::image::ImageDataType::RGBA, 
            speedy2d::image::ImageSmoothingMode::Linear, 
            self.texture_size, 
            &self.texture_slice
        ).unwrap();

        graphics.draw_rectangle_image(self.map_rect.clone(), &img);
        
        graphics.draw_text(
            (10.0, 10.0), Color::WHITE, 
            &self.font.layout_text(&(1000 / &self.delta.elapsed().as_millis()).to_string(), 32.0, TextOptions::new())
        );

        helper.request_redraw();
    }
}

fn main() {
    let window = Window::new_centered("Lenia", (DISPLAY_RES.0, DISPLAY_RES.1)).unwrap();
    window.run_loop(MyWindowHandler::new(SEED));
}





// Legacy code

        
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