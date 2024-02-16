use std::ops;

#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,   
    pub cols: usize,
    pub data: Vec<f64>
}

impl Matrix{
    pub fn new(rows:usize, cols:usize) -> Matrix {
        let data = vec![0.0; rows*cols];
        Matrix{rows, cols, data}
    }
}

// Matrix multiplication
impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, second:Matrix) -> Matrix {
        if self.cols != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = Matrix::new(self.rows, second.cols);
        for i in 0..self.rows {
            for j in 0..second.cols {
                let mut sum:f64 = 0.0;
                for k in 0..second.rows {
                    sum += self.data[i*self.cols +k] *  second.data[k*second.cols + j];
                }
                result.data[i*second.cols +j] = sum;
            }
        }
        result
    }
}

// Matrix addition
impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;
    
    fn add(self, second:Matrix) -> Matrix {
        if self.cols != second.cols || self.rows != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i]+second.data[i];
        }
        result
    }
}

// Multiplication of Matrix by a scalar
impl ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar:f64)-> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        if scalar == 0.0 {
            return result;
        }
        for i in 0..self.data.len() {
            result.data[i] = self.data[i]*scalar;
        }
        result
    }
}
