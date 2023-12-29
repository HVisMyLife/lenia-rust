#![allow(clippy::ptr_arg)]
use rayon::prelude::*;
use ndarray::{prelude::*, Zip};
use ndarray_conv::*;
use macroquad::prelude::*;
use bincode::{serialize, deserialize};
use std::fs::File;
use std::io::prelude::*;

mod fta;
use fta::FrameTimeAnalyzer;


// size of map for convolution, it is later padded for warp func, so we have to cut it for now
pub const MAP_SIZE: (i32, i32) = (1280, 820);
pub const SEED: u64 = 1;

struct Layer {
    size: i32,
    kernel: Array2<f32>,
    pub g_params: (f32,f32,f32), // dt, g-center, g-width ( 0-100 neighbourhood )
}

struct Ecosystem {
    // texture for drawing states
    texture_slice: Vec<u8>,
    texture_size: [u16; 2],
    
    // main array of states
    pub map: Array2<f32>,

    // layers
    pub layer: [Layer; 2],

    pub fitness: f32, // 0 - 2 - 20
    pub cycles: u32, // 0 - 2 - 20
}

impl Ecosystem {
    fn new() -> Self {

        // generate random starting map
        let map = Array2::<f32>::zeros((MAP_SIZE.0 as usize, MAP_SIZE.1 as usize));
        // for y in rand::gen_range((MAP_SIZE.1 as f32/3.0) as i32, MAP_SIZE.1/2)..rand::gen_range(MAP_SIZE.1/2, (MAP_SIZE.1 as f32/1.5) as i32) {
        //     for x in rand::gen_range((MAP_SIZE.0 as f32/10.0) as i32, MAP_SIZE.0/2)..rand::gen_range(MAP_SIZE.0/2, (MAP_SIZE.0 as f32/1.1) as i32) {
        //         map[[x as usize, y as usize]] = rand::gen_range(0.0, 1.0);
        //     }
        // } 

        let mut layer: [Layer; 2] = [
            Layer {
                size: 92,
                kernel: Default::default(),
                g_params: (0.100, 0.118, 0.013),
            },
            Layer {
                size: 92,
                kernel: Default::default(),
                g_params: (0.100, 0.167, 0.022),
            },
        ];
        layer[0].kernel = Array2::<f32>::zeros(( (2*layer[0].size+1) as usize, (2*layer[0].size+1) as usize ));
        layer[1].kernel = Array2::<f32>::zeros(( (2*layer[1].size+1) as usize, (2*layer[1].size+1) as usize ));
        // generate kernel 0
        for ny in -layer[0].size..=layer[0].size {
            let d = f32::sqrt((layer[0].size * layer[0].size - ny * ny) as f32) as i32;
            for nx in -layer[0].size..=layer[0].size {
                let r = (f32::sqrt((nx.pow(2) + ny.pow(2)) as f32) + 1.0) / layer[0].size as f32;
    
                if ny == 0 && nx == 0 || nx < -d || nx > d { layer[0].kernel[[(nx+layer[0].size)as usize, (ny+layer[0].size)as usize]] = 0.0;}
                else {layer[0].kernel[[(nx+layer[0].size)as usize, (ny+layer[0].size)as usize]] = ( -((r - 0.5)/0.15).powi(2) / 2.0 ).exp();}// kernel shape
            }
        }
        layer[0].kernel /= layer[0].kernel.sum();

        // generate kernel 1
        for ny in -layer[1].size..=layer[1].size {
            let d = f32::sqrt((layer[1].size * layer[1].size - ny * ny) as f32) as i32;
            for nx in -layer[1].size..=layer[1].size {
                let r = (f32::sqrt((nx.pow(2) + ny.pow(2)) as f32) + 1.0) / layer[0].size as f32;
    
                if ny == 0 && nx == 0 || nx < -d || nx > d { layer[1].kernel[[(nx+layer[1].size)as usize, (ny+layer[1].size)as usize]] = 0.0;}
                else {layer[1].kernel[[(nx+layer[1].size)as usize, (ny+layer[1].size)as usize]] =
                    ( ( -150. * (r-(3./4.)).powi(2) ).exp() * (1./3.) ) +
                    ( ( -150. * (r-(1./2.)).powi(2) ).exp() * (2./3.) ) +
                    ( ( -150. * (r-(1./4.)).powi(2) ).exp() * (1./1.) )
                    ;}// kernel shape

            }
        }
        layer[1].kernel /= layer[1].kernel.sum();
        
        Self { 

            texture_slice: [255;  (4 * MAP_SIZE.1 * MAP_SIZE.0) as usize].to_vec(),
            texture_size: [MAP_SIZE.0 as u16, MAP_SIZE.1 as u16],

            map,
            layer,
            fitness: 0.,
            cycles: 0,
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

    fn calc_conv(&mut self) {
        // fft convolving map using kernel
        let mut mc0: Array2<f32> = Default::default();
        let mut mc1: Array2<f32> = Default::default();
        rayon::join(
            || {mc0 = self.map.conv_2d_fft(&self.layer[0].kernel, PaddingSize::Same, PaddingMode::Circular).unwrap();}, 
            || {mc1 = self.map.conv_2d_fft(&self.layer[1].kernel, PaddingSize::Same, PaddingMode::Circular).unwrap();}
        );
        
        // applying growth mapping function
        Zip::from(&mut self.map)
            .and(&mc0)
            .and(&mc1)
            .par_for_each(|m, &o0, &o1| {
                *m = (
                    *m + 
                    (self.layer[0].g_params.0 * (( -((o0 -self.layer[0].g_params.1) / self.layer[0].g_params.2).powi(2) / 2.0 ).exp() * 2.0 -1.0) + 
                        self.layer[1].g_params.0 * (( -((o1 -self.layer[1].g_params.1) / self.layer[1].g_params.2).powi(2) / 2.0 ).exp() * 2.0 -1.0)
                    ) / 2.
                ).clamp(0.0, 1.0)     // clamping between 0-1: A + dtG(A*K)
            });

        self.fitness = (self.map.sum()/(MAP_SIZE.0 * MAP_SIZE.1)as f32*10000.).round() / 100.;
        self.cycles+=1;
    }
   
    fn on_draw(&mut self, pause: &bool){
        clear_background(Color::from_rgba(24, 24, 24, 255));

        if !pause {
        //calculating convolution
            self.calc_conv();
        // calculating img pixels
            self.calc_img();
        }

        // generating texture
        let tx = Texture2D::from_rgba8(self.texture_size[0], self.texture_size[1], &self.texture_slice);
        draw_texture(&tx, 0., 0., WHITE);
    }
}

struct Best {
    pub g_params: [(f32,f32,f32); 2], // dt, g-center, g-width ( 0-100 neighbourhood )
    pub cycles: u32,
}

impl Best {
    pub fn compare(&mut self, eco: &mut Ecosystem) {
        if eco.fitness > 0.1 {eco.cycles = (eco.cycles as f32 * 0.8) as u32;} //overflow punish
        if eco.cycles > self.cycles {
            self.g_params[0] = eco.layer[0].g_params;
            self.g_params[1] = eco.layer[1].g_params;
            self.cycles = eco.cycles;
        }
        match rand::gen_range(0, 4) {
            0 => eco.layer[0].g_params.1 = self.g_params[0].1 + rand::gen_range(-0.01, 0.01),
            1 => eco.layer[0].g_params.2 = self.g_params[0].2 + rand::gen_range(-0.001, 0.001),
            2 => eco.layer[1].g_params.1 = self.g_params[1].1 + rand::gen_range(-0.01, 0.01),
            3 => eco.layer[1].g_params.2 = self.g_params[1].2 + rand::gen_range(-0.001, 0.001),
            _ => {},
        }
        eco.map = load("data.bin");
        eco.cycles = 0;
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(SEED);
    let mut eco = Ecosystem::new();
    eco.map = load("data.bin");
    let font = macroquad::text::load_ttf_font("font.ttf").await.unwrap();
    let mut ui = UI::new(font);

    loop {
        eco.on_draw(&ui.pause);
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
    fn new(font: Font) -> Self {
        
        UI { 
            fta: FrameTimeAnalyzer::new(16), 
            selection: 0, 
            pause: false, 
            autotune: true,
            best: Best {
                g_params: [(0.,0.,0.), (0.,0.,0.)],
                cycles: 0,
            },
            font,
        }
    }
    fn execute(&mut self, eco: &mut Ecosystem) -> bool {
        if self.autotune && (eco.fitness == 0. || eco.fitness > 5.0) {self.best.compare(eco);}
        if is_key_pressed(KeyCode::T) {self.autotune = !self.autotune;} // add to best
        if is_key_pressed(KeyCode::G) {self.best.cycles = 0;} // add to best
        if is_key_pressed(KeyCode::F) {eco.cycles = 0; eco.fitness = 50.} // add to best
        if is_key_pressed(KeyCode::A) {self.pause = !self.pause;}
        if is_key_pressed(KeyCode::S) {save("data.bin", &eco.map);}
        if is_key_pressed(KeyCode::L) {eco.map = load("data.bin");}
        if is_key_pressed(KeyCode::Up) {self.selection-=1;}
        if is_key_pressed(KeyCode::Down) {self.selection+=1;}
        self.selection = self.selection.clone().clamp(0, 5);
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

// save nn to file
fn save(path: &str, data: &Array2<f32>) {        
    // convert simplified nn to Vec<u8>
    let encoded: Vec<u8> = serialize(
        data
    ).unwrap();
 
    // open file and write whole Vec<u8>
    let mut file = File::create(path).unwrap();
    file.write_all(&encoded).unwrap();
} 
 
// load nn from file
fn load(path: &str) -> Array2<f32> {
    // convert readed Vec<u8> to plain nn
    let mut buffer = vec![];
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    let decoded: Array2<f32> = deserialize(&buffer).unwrap();
 
    decoded
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
