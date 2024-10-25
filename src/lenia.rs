#![allow(dead_code)]
use ndarray::prelude::*;
use ndarray_conv::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    GaussianBump, // width, offset
    TripleBump,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Destiny {
    Kernel,
    GrowthMap,
    Misc
}

// Kernel and growth functions are the same, only diffrence is that, growth function x changes with
// pi*r^2. Delta can be applied later
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub radius: usize,
    pub shape: Shape,
    pub centering: bool,  // should it be centered at x (moved down)
    pub parameters: Vec<f32>,  // it will be clamped to 0<>1
    pub destiny: Destiny
}

impl Function {
    pub fn new(radius: usize, shape: Shape, centering: bool, parameters: Option<Vec<f32>>, destiny: Option<Destiny>) -> Self {
        Function {
            radius,
            shape,
            centering,
            parameters: parameters.unwrap_or(vec![]),
            destiny: destiny.unwrap_or(Destiny::Misc),
        }
    }

    pub fn calc(&self, x: f32) -> f32 {
        let mut x = x;
        let mut y;

        match self.destiny {
            Destiny::Kernel => {
                x /= self.radius as f32; 
                if x > 1. {return 0.;} // it's outside circle region 
            }, // x must be 0<>1
            Destiny::GrowthMap => {}// normalizing kernel below x /= self.radius.pow(2) as f32 * PI, // square cube law
            Destiny::Misc => {}
        }
        
        match self.shape {
            Shape::GaussianBump => {
                y = ( -( ( x - self.parameters[0]) / self.parameters[1]).powi(2) / 2.0 ).exp();
            },
            Shape::TripleBump => {
                y = ( ( -150. * (x-(3./4.)).powi(2) ).exp() * (1./3.) ) +
                ( ( -150. * (x-(1./2.)).powi(2) ).exp() * (2./3.) ) +
                ( ( -150. * (x-(1./4.)).powi(2) ).exp() * (1./1.) );
            }
        }

        y = y.clamp(0., 1.);
        if self.centering { y -= 0.5; }

        y
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub kernel: Function,
    kernel_lookup: Array2<f32>,
    pub growth_map: Function,
    pub channel_id: usize, // number of channel that will be used as input
    matrix_out: Array2<f32>
}

impl Layer {
    pub fn new(
        kernel: Function,
        growth_map: Function,
        channel_id: usize
    ) -> Self {
        let r = kernel.radius;
        Layer { 
            kernel, kernel_lookup: Array2::<f32>::zeros((r * 2 + 1, r * 2 + 1)),
            growth_map, channel_id, matrix_out: Default::default()
        }
    }

    fn generate_kernel_lookup(&mut self) {
        for x in -(self.kernel.radius as i64)..=self.kernel.radius as i64 {
            for y in -(self.kernel.radius as i64)..=self.kernel.radius as i64 {
                let r = ( (x*x+y*y) as f32 ).sqrt();
                self.kernel_lookup[[(x+self.kernel.radius as i64) as usize, (y+self.kernel.radius as i64) as usize]] = self.kernel.calc(r);
            }    
        }
        self.kernel_lookup /= self.kernel_lookup.sum(); // no matter kernel radius, sum off ideal
        // convolution will always be equal to 1
    }

    fn run(&mut self, channel: &Channel) {
        // convoluted matrix
        self.matrix_out = channel.matrix.conv_2d_fft(&self.kernel_lookup, PaddingSize::Same, PaddingMode::Circular).unwrap();
        self.matrix_out.par_map_inplace(|x|{
            *x = self.growth_map.calc(*x);
        });
    }
}

// Array3 ????? Could be xddd
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    pub matrix: Array2<f32>,
    matrix_out: Array2<f32>,
    layer_counter: usize,
}

impl Channel {
    pub fn new(matrix: Array2<f32>) -> Self {
        Self { 
            matrix_out: Array2::<f32>::zeros(matrix.dim()),
            matrix, 
            layer_counter: 0,
        }
    }

    // things to do after layer computation
    fn finish(&mut self, delta: f32) {
        self.matrix_out /= self.layer_counter as f32 ;  // change is divided by amount of layers
        self.matrix_out *= delta;   // incorporate delta
        self.layer_counter = 0;
        ndarray::Zip::from(&mut self.matrix).and(&mut self.matrix_out).par_for_each(|m, m_out|{
            *m = (*m + *m_out).clamp(0., 1.);  // add corrections to matrix
            *m_out = 0.;  // zero corrections for next turn
        });
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Eco {
    pub channels: Vec<Channel>,
    pub layers: Vec<Layer>,
    pub delta: f32,
    pub size: (usize, usize),
    pub cycles: usize,
    pub fitness: f32,
}

impl Eco {
    pub fn new(size: (usize, usize), delta: f32, cycles: usize, channels: Vec<Channel>, layers: Vec<Layer>) -> Self {

        Self { channels, layers, 
            delta, size, 
            cycles, fitness: 0.
        }
    }

    pub fn init(&mut self) {
        self.layers.par_iter_mut().for_each(|l|{
            l.generate_kernel_lookup();
        });
    }

    pub fn evaluate(&mut self) {
        self.layers.par_iter_mut().for_each(|l|{
            l.run(&self.channels[l.channel_id]);
        });

        self.channels.par_iter_mut().for_each(|ch|{
            self.layers.iter().for_each(|l|{
                ch.layer_counter += 1;
                ch.matrix_out.scaled_add(1., &l.matrix_out); // add to output matrix
            });
            ch.finish(self.delta);
        });

        self.cycles += 1;

        // calculate fitness
        self.channels.iter().for_each(|ch|{
            self.fitness += ch.matrix_out.mean().unwrap();
        });
        self.fitness /= self.channels.len() as f32;
    }

}
