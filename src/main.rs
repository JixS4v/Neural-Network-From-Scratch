use nanorand::{Rng, WyRand};
use std::ops;

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
    pub fn normalize(&self) {
        let sum: f64 = self.data.iter().sum();
        let mut normalized = self.clone();
        normalized.data = normalized.data.iter().map(|x| x / sum).collect();
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
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] += second.data[i];
        }
        result
    }
}

impl ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, second: Matrix) -> Matrix {
        if self.cols != second.cols || self.rows != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] -= second.data[i];
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

impl ops::Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, second: &Matrix) -> Matrix {
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

pub struct NetworkState {
    input_neurons: Matrix,
    hidden_layer_neurons: Vec<Matrix>,
    output_neurons: Matrix,
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
    // Returns only the output neurons with the input neurons
    pub fn quick_infer(&self, input: Matrix, activation: fn(&f64) -> f64) -> Result<Matrix, &str> {
        if input.rows != self.input_size {
            return Err("Input size mismatch");
        }
        let mut values: Matrix = input.clone();
        for i in 0..self.hidden_layers.len() {
            values =
                values * self.hidden_layers[i].weights.clone() + self.hidden_layers[i].bias.clone();
            // We activate
            values.data = values.data.iter().map(activation).collect();
        }
        values = values * self.output.weights.clone() + self.output.bias.clone();
        values.data = values.data.iter().map(activation).collect();
        Ok(values)
    }
    pub fn forward_propagate(
        &self,
        input: &Matrix,
        activation: fn(&f64) -> f64,
    ) -> Result<NetworkState, &str> {
        if input.rows != self.input_size {
            return Err("Input size mismatch");
        }
        let mut hidden_layer_neurons: Vec<Matrix> = Vec::new();
        for i in 0..self.hidden_layers.len() {
            hidden_layer_neurons
                .push(input * &self.hidden_layers[i].weights + self.hidden_layers[i].bias.clone());
            hidden_layer_neurons[i].data = hidden_layer_neurons[i]
                .data
                .iter()
                .map(activation)
                .collect();
        }
        let hidden_neurons_last_index = hidden_layer_neurons.len() - 1;
        let mut output_neurons = &hidden_layer_neurons[hidden_neurons_last_index]
            * &self.output.weights
            + self.output.bias.clone();
        output_neurons.data = output_neurons.data.iter().map(activation).collect();
        return Ok(NetworkState::new(
            input.clone(),
            hidden_layer_neurons,
            output_neurons,
        ));
    }
}

impl NetworkState {
    pub fn new(
        input_neurons: Matrix,
        hidden_layer_neurons: Vec<Matrix>,
        output_neurons: Matrix,
    ) -> NetworkState {
        NetworkState {
            input_neurons,
            hidden_layer_neurons,
            output_neurons,
        }
    }

    // Calculating the gradient by backpropagation
    pub fn backpropagate(&self, network: Network) -> Vec<Matrix> {}
}

// Rectified Linear Unit function for activation
fn relu(value: &f64) -> f64 {
    let value = *value;
    if value > 0.0 {
        value
    } else {
        0.0
    }
}

fn squared_error(input: Matrix, output: Matrix) -> Matrix {
    let mut cost = output - input;
    cost.data = cost.data.iter().map(|x| (*x).powf(2.0)).collect();
    cost
}

fn main() {
    let mut network = Network::new(28 * 28, vec![16, 16], 10);
    network.randomize();
}
