use nanorand::{Rng, WyRand};
use std::ops;
use std::cmp;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let data = vec![0.0; rows * cols];
        Matrix { rows, cols, data }
    }
    pub fn randomize(&mut self, min: f64, max: f64) {
        // Bad code to get around the int limitation of nanorand
        let min_int: i32 = (min * 100.0) as i32;
        let max_int: i32 = (max * 100.0) as i32;
        let mut rng = WyRand::new();
        self.data = self
            .data
            .iter()
            .map(|_| (rng.generate_range(min_int..=max_int) as f64) / 100.0)
            .collect();
    }
}

// Matrix multiplication
impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, second: Matrix) -> Matrix {
        if self.cols != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = Matrix::new(self.rows, second.cols);
        for i in 0..self.rows {
            for j in 0..second.cols {
                let mut sum: f64 = 0.0;
                for k in 0..second.rows {
                    sum += self.data[i * self.cols + k] * second.data[k * second.cols + j];
                }
                result.data[i * second.cols + j] = sum;
            }
        }
        result
    }
}

// Matrix addition
impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, second: Matrix) -> Matrix {
        if self.cols != second.cols || self.rows != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + second.data[i];
        }
        result
    }
}

// Multiplication of Matrix by a scalar
impl ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        if scalar == 0.0 {
            return result;
        }
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * scalar;
        }
        result
    }
}
pub struct Layer {
    pub weights: Matrix,
    pub bias: Matrix,
}

impl Layer {
    pub fn new(neurons: usize, weights: usize) -> Layer {
        Layer {
            weights: Matrix::new(neurons, weights),
            bias: Matrix::new(neurons, 1),
        }
    }
}

pub struct Network {
    input_size: usize,
    hidden_layers: Vec<Layer>,
    output: Layer,
}

impl Network {
    pub fn new(input_size: usize, hidden_layer_sizes: Vec<usize>, output_size: usize) -> Network {
        let mut hidden_layers: Vec<Layer> = Vec::new();
        for i in 0..hidden_layer_sizes.len() {
            if i == 0 {
                hidden_layers.push(Layer::new(hidden_layer_sizes[i], input_size));
                continue;
            }
            hidden_layers.push(Layer::new(hidden_layer_sizes[i], hidden_layer_sizes[i - 1]));
        }
        Network {
            input_size,
            hidden_layers,
            output: Layer::new(
                output_size,
                hidden_layer_sizes[hidden_layer_sizes.len() - 1],
            ),
        }
    }
    pub fn randomize(&mut self) {
        for i in 0..self.hidden_layers.len() {
            self.hidden_layers[i].weights.randomize(-1.0, 1.0); // Default weight range is [-1,1]
            self.hidden_layers[i].bias.randomize(-10.0, 10.0); // Default bias range is [-10,10]
        }
        self.output.weights.randomize(-1.0, 1.0);
        self.output.bias.randomize(-10.0, 10.0);
    }
    pub fn propagate(&self, input:  Matrix, activation: fn(&f64)->f64) -> Result<Matrix, &str> {
        if input.rows != self.input_size {
            return Err("Input size mismatch");
        }
        let mut values: Matrix = input.clone(); 
        for i in 0..self.hidden_layers.len() {
            values = values*self.hidden_layers[i].weights.clone() + self.hidden_layers[i].bias.clone();
            // We activate
            values.data = values.data.iter().map(activation).collect();
        }
        values = values*self.output.weights.clone() + self.output.bias.clone();
        values.data = values.data.iter().map(activation).collect();
        Ok(values)
    }
}

fn main() {
    let mut network = Network::new(28 * 28, vec![16, 16], 10);
    network.randomize();
}
