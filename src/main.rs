#![allow(clippy::ptr_arg)]
use rayon::prelude::*;
use ndarray::{prelude::*, Zip};
use ndarray_conv::*;
use macroquad::prelude::*;

pub const K_R: i32 = 60;            // radius of kernel no.0 20
pub const K_MAX: i32 = K_R;         // maximum radius of kernel any

pub const DT: f32 = 10.0;       // % add each cycle
pub const CENTER: f32 = 15.00;  // % of neighbours full .14
pub const WIDTH: f32 = 1.42;    // width .03

// size of map for convolution, it is later padded for warp func, so we have to cut it for now
pub const MAP_SIZE: (i32, i32) = (1024-2*K_R, 1024-2*K_R);
pub const SEED: u64 = 2;

pub const DISPLAY_SCALE: f32 = 1.0;


struct Ecosystem {
    // main lenia parameters
    g_params: (f32,f32,f32), // dt, g-center, g-width ( 0-100 neighbourhood )
    // precalculated kernel values
    kernel: Array2<f32>,
    
    // texture for drawing states
    texture_slice: Vec<u8>,
    texture_size: [u16; 2],
    
    // main array of states
    map: Array2<f32>,     
    map_convolved: Array2<f32>,
}

impl Ecosystem {
    fn new() -> Self {

        // generate random starting map
        let mut map = Array2::<f32>::zeros((MAP_SIZE.0 as usize, MAP_SIZE.1 as usize));
        for y in rand::gen_range((MAP_SIZE.1 as f32/3.0) as i32, MAP_SIZE.1/2)..rand::gen_range(MAP_SIZE.1/2, (MAP_SIZE.1 as f32/1.5) as i32) {
            for x in rand::gen_range((MAP_SIZE.0 as f32/10.0) as i32, MAP_SIZE.0/2)..rand::gen_range(MAP_SIZE.0/2, (MAP_SIZE.0 as f32/1.1) as i32) {
                map[[x as usize, y as usize]] = rand::gen_range(0.0, 1.0);
            }
        } 

        // generate kernel
        let mut kernel: Array2<f32> = Array2::<f32>::zeros(( (2*K_R+1) as usize, (2*K_R+1) as usize ));
        for ny in -K_R..=K_R {
            let d = f32::sqrt((K_R * K_R - ny * ny) as f32) as i32;
            for nx in -K_R..=K_R {
                let r = (f32::sqrt((nx.pow(2) + ny.pow(2)) as f32) + 1.0) / K_R as f32;
    
                if ny == 0 && nx == 0 || nx < -d || nx > d { kernel[[(nx+K_R)as usize, (ny+K_R)as usize]] = 0.0;}
                else {kernel[[(nx+K_R)as usize, (ny+K_R)as usize]] = ( -((r - 0.5)/0.15).powi(2) / 2.0 ).exp();}// kernel shape
            }
        }
        kernel /= kernel.sum();

        
        Self { 
            g_params: (DT/100.0, CENTER / 100.0, WIDTH / 100.0), // 0-100 neighbourhood
            kernel,

            texture_slice: [255;  (4 * MAP_SIZE.1 * MAP_SIZE.0) as usize].to_vec(),
            texture_size: [MAP_SIZE.0 as u16, MAP_SIZE.1 as u16],

            map, 
            map_convolved: Array2::<f32>::zeros((MAP_SIZE.0 as usize, MAP_SIZE.1 as usize)),
        }
    }

    // convert cell value to color spectrum
    fn calc_img(&mut self) {
        self.texture_slice.par_chunks_mut(4).enumerate().for_each(|(i, x)| {
            let col = (self.map[[i%MAP_SIZE.0 as usize, i/MAP_SIZE.0 as usize]] * 255.0) as i32;
            
            x[0] = (-(col/4 - 16).pow(2) + 255).clamp(0, 255) as u8; // 3-16
            x[1] = (-(col/4 - 32).pow(2) + 255).clamp(0, 255) as u8; // 3-44
            x[2] = (-(col/4 - 48).pow(2) + 255).clamp(0, 255) as u8; // 3-72
            //x[3] = (col*2).clamp(0, 255) as u8;
            if col > 0 {x[3] = 255;} else {x[3] = 0;}
        });
    }
   
    fn on_draw(&mut self){
        clear_background(Color::from_rgba(24, 24, 24, 255));

        // fft convolving map using kernel
        self.map_convolved = self.map.conv_2d_fft(&self.kernel, PaddingSize::Same, PaddingMode::Circular).unwrap().clone();
        
        // applying growth mapping function
        Zip::from(&mut self.map)
            .and(&self.map_convolved)
            .par_for_each(|m, &o| {
                *m = (
                    *m 
                    + self.g_params.0 * (( -((o-self.g_params.1) / self.g_params.2).powi(2) / 2.0 ).exp() * 2.0 -1.0)
                ).clamp(0.0, 1.0)     // clamping between 0-1: A + dtG(A*K)
            });

        // calculating img pixels
        self.calc_img();

        // generating texture
        let tx = Texture2D::from_rgba8(self.texture_size[0], self.texture_size[1], &self.texture_slice);
        draw_texture(&tx, 0., 0., WHITE);
    }
}



#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(SEED);
    let mut eco = Ecosystem::new();

    loop {
        if is_key_down(KeyCode::Q) {break;}
        eco.on_draw();
        draw_text(&get_fps().to_string(), 5., 35., 35., WHITE);
        next_frame().await
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Lenia".to_owned(),
        fullscreen: false,
        window_width: (MAP_SIZE.0 as f32 * DISPLAY_SCALE) as i32,
        window_height: (MAP_SIZE.1 as f32 * DISPLAY_SCALE) as i32,
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
