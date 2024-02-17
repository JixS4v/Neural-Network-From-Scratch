mod matrix;
use crate::matrix::Matrix;

pub struct Layer {
    pub neurons: Matrix,
    pub weights: Matrix,
    pub bias: Matrix,
}

impl Layer {
    pub fn new(neurons: usize, weights: usize) -> Layer {
        Layer {
            neurons: Matrix::new(1, neurons),
            weights: Matrix::new(weights, neurons),
            bias: Matrix::new(1, neurons),
        }
    }
}
