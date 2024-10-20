use rayon::prelude::*;
use ndarray::{prelude::*, Zip};
use ndarray_conv::*;
use rand::Rng;

pub struct Ecosystem {
    // texture for drawing states
    texture_slice: Vec<u8>,
    texture_size: [u16; 2],
    
    // main array of states
    pub map: Array2<f32>,

    // layers
    pub layer: [Layer; 2],

    pub fitness: f32, // 0 - 2 - 20
    pub cycles: u32, // 0 - 2 - 20

    map_size: (usize, usize),
}

impl Ecosystem {
    pub fn new(map_size: (usize, usize)) -> Self {

        // generate random starting map
        let map = Array2::<f32>::zeros((map_size.0, map_size.1));
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

            map_size,
            texture_slice: vec![255;  4 * map_size.1 * map_size.0],
            texture_size: [map_size.0 as u16, map_size.1 as u16],

            map,
            layer,
            fitness: 0.,
            cycles: 0,
        }
    }

    // convert cell value to color spectrum
    fn calc_img(&mut self) {
        self.texture_slice.par_chunks_mut(4).enumerate().for_each(|(i, x)| {
            let col = (self.map[[i%self.map_size.0, i/self.map_size.0]] * 255.0) as i32;
            
            x[0] = (-(col/4 - 16).pow(2) + 255).clamp(0, 255) as u8; // 3-16
            x[1] = (-(col/4 - 32).pow(2) + 255).clamp(0, 255) as u8; // 3-44
            x[2] = (-(col/4 - 48).pow(2) + 255).clamp(0, 255) as u8; // 3-72
            //x[3] = (col*2).clamp(0, 255) as u8;
            if col > 0 {x[3] = 255;} else {x[3] = 0;}
        });
    }

    // calculate convolution while applying growth map funct
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

        self.fitness = (self.map.sum()/(self.map_size.0 * self.map_size.1)as f32*10000.).round() / 100.;
        self.cycles+=1;
    }
   
    pub fn run(&mut self, pause: &bool) -> (&u16, &u16, &Vec<u8>) {
        if !pause { 
            self.calc_conv();
            self.calc_img();
        }

        // generating texture
        (&self.texture_size[0], &self.texture_size[1], &self.texture_slice)
    }
}



pub struct Layer {
    pub size: i32,
    kernel: Array2<f32>,
    pub g_params: (f32,f32,f32), // dt, g-center, g-width ( 0-100 neighbourhood )
}


pub struct Best {
    pub g_params: [(f32,f32,f32); 2], // dt, g-center, g-width ( 0-100 neighbourhood )
    pub cycles: u32,
}

impl Best {
    pub fn compare(&mut self, eco: &mut Ecosystem) -> bool {
        let is_better;
        if eco.fitness > 0.1 {eco.cycles = (eco.cycles as f32 * 0.8) as u32;} //overflow punish
        if eco.cycles > self.cycles {
            self.g_params[0] = eco.layer[0].g_params;
            self.g_params[1] = eco.layer[1].g_params;
            self.cycles = eco.cycles;
            is_better = true;
        }
        else {
            is_better = false;
        }
        match rand::thread_rng().gen_range(0..4) {
            0 => eco.layer[0].g_params.1 = self.g_params[0].1 + rand::thread_rng().gen_range(-0.01..0.01),
            1 => eco.layer[0].g_params.2 = self.g_params[0].2 + rand::thread_rng().gen_range(-0.001..0.001),
            2 => eco.layer[1].g_params.1 = self.g_params[1].1 + rand::thread_rng().gen_range(-0.01..0.01),
            3 => eco.layer[1].g_params.2 = self.g_params[1].2 + rand::thread_rng().gen_range(-0.001..0.001),
            _ => {},
        }
        eco.cycles = 0;
        is_better
    }
}


