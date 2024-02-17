mod layer;
mod matrix;
use crate::matrix::Matrix;
use crate::layer::Layer;

pub struct Network {
    input: Matrix,
    hidden_layers: Vec<Layer>,
    output: Layer,
}

impl Network {
    pub fn new(input_size: usize, hidden_layer_sizes: Vec<usize>, output_size: usize) {
        let mut hidden_layers: Vec<Layer> = Vec::new();
        for i in 0..hidden_layer_sizes.len() {
            if i == 0 {
                hidden_layers.push(Layer::new(hidden_layer_sizes[i], input_size));
                continue;
            }
            hidden_layers.push(Layer::new(
                hidden_layer_sizes[i],
                hidden_layer_sizes[i - 1],
            ));
        }
        Network {
            input: Matrix::new(1, input_size),
            hidden_layers: hidden_layers,
            output: Layer::new(
                output_size,
                hidden_layer_sizes[hidden_layer_sizes.len() - 1],
            ),
        }
    }
    pub fn randomize(&mut self) {
        for i in 0..self.hidden_layers.len() {
            self.hidden_layers[i].weights.data.randomize(-1, 1); // Default weight range is [-1,1]
            self.hidden_layers[i].bias.data.randomize(-10, 10); // Default bias range is [-10,10]
        }
        self.output.weights.data.randomize(-1, 1);
        self.output.bias.data.randomize(-10, 10);
    }
}
