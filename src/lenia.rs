#![allow(dead_code)]
use ndarray::prelude::*;
use ndarray_conv::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strum::Display;


pub trait Cycle {
    fn next(&mut self) -> Self;
    fn previous(&mut self) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Display)]
pub enum Shape {
    GaussianBump, // width, offset
    ExponentialDecay,
    SmoothTransition,
    MexicanHat,
    TripleBump,
}

impl Cycle for Shape {
    fn next(&mut self) -> Self {
        match self {
            Shape::GaussianBump => Shape::ExponentialDecay,
            Shape::ExponentialDecay => Shape::SmoothTransition,
            Shape::SmoothTransition => Shape::MexicanHat,
            Shape::MexicanHat => Shape::TripleBump,
            Shape::TripleBump => Shape::GaussianBump,
        }
    }
    fn previous(&mut self) -> Self {
        match self {
            Shape::GaussianBump => Shape::TripleBump,
            Shape::ExponentialDecay => Shape::GaussianBump,
            Shape::SmoothTransition => Shape::ExponentialDecay,
            Shape::MexicanHat => Shape::SmoothTransition,
            Shape::TripleBump => Shape::MexicanHat,
        }
    }
}

// Kernel and growth functions are the same, only diffrence is that, growth function x changes with
// pi*r^2. Delta can be applied later
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub shape: Shape,
    pub centering: bool,  // should it be centered at x (moved down)
    pub parameters: Vec<f32>,  // it will be clamped to 0<>1
}

impl Function {
    pub fn new(shape: Shape, centering: bool, parameters: Vec<f32>) -> Self {
        Function {
            shape,
            centering,
            parameters,
        }
    }

    pub fn calc(&self, x: f32) -> f32 {
        let mut y;

        // 0 - width, 1 - offset
        match self.shape {
            Shape::GaussianBump => {
                y = ( -( ( x - self.parameters[1]) / self.parameters[0] ).powi(2) / 2. ).exp();
            },
            Shape::ExponentialDecay => { // comes from infinity, so have to be clamped
                y = ( -( ( x - self.parameters[1]) / self.parameters[0] ) ).exp().clamp(0., 1.);
            },
            Shape::SmoothTransition => {
                y = 1. / ( 1. + ( ( x - self.parameters[1]) / self.parameters[0] ).exp() );
            },
            Shape::MexicanHat => {
                y = 1. / ( 1. + ( ( ( ( x - self.parameters[1] ) / self.parameters[0] ).powi(2) - 1. ).powi(2) ) );
            },
            Shape::TripleBump => { // when wide goes to 2
                y = ( 0.6 * ( -( ( x - self.parameters[1] + 0.25 ) / self.parameters[0] ).powi(2) ).exp() ) +
                    ( 0.8 * ( -( ( x - self.parameters[1] + 0.00 ) / self.parameters[0] ).powi(2) ).exp() ) +
                    ( 0.6 * ( -( ( x - self.parameters[1] - 0.25 ) / self.parameters[0] ).powi(2) ).exp() ).clamp(0., 1.);
            },
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
    matrix_out: Array2<f32>,
    pub radius: usize,
}

impl Layer {
    pub fn new(
        kernel: Function,
        growth_map: Function,
        channel_id: usize,
        radius: usize
    ) -> Self {
        Layer { 
            kernel, kernel_lookup: Array2::<f32>::zeros((radius * 2 + 1, radius * 2 + 1)),
            growth_map, channel_id, matrix_out: Default::default(), radius
        }
    }

    fn generate_kernel_lookup(&mut self) {
        for x in -(self.radius as i64)..=self.radius as i64 {
            for y in -(self.radius as i64)..=self.radius as i64 {
                let r = ( (x*x+y*y) as f32 ).sqrt();
                self.kernel_lookup[[(x+self.radius as i64) as usize, (y+self.radius as i64) as usize]] 
                    = self.kernel.calc(r/self.radius as f32);
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
        self.fitness /= self.channels.len() as f32 * 10000.;
    }

}
