mod Matrix

#[derive(Debug)]
pub struct Layer {
    pub neurons: Matrix,
    pub weights: Matrix,
    pub bias: Matrix
}

impl Layer {
    fn new(neurons, weights) -> Layer {
        Layer {
            neurons: Matrix::new(1, neurons),
            weights: Matrix::new(weights, neurons),
            bias: Matrix::new(1, neurons)
        }
    }
