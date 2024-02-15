pub struct Matrix {
    rows: usize,   
    cols: usize,
    data: Vec<f64>
}
impl Matrix{
    fn new(rows:usize, cols:usize) -> Matrix {
        let data = vec![0; rows*cols];
        Matrix{rows, cols, data}
    }
}

// Matrix multiplication
impl ops::Mul<Matrix> for Matrix {
    fn mul(self, Matrix:second) -> Matrix {
        if self.cols != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = Matrix::new(self.rows, second.cols);
        for i in 0..self.rows {
            for j in 0..second.cols {
                let mut sum:f64 = 0;
                for k in 0..second.rows {
                    sum += self.data[i*self.cols +k] *  second.data[k*second.cols + j];
                }
                result.data[i*second.cols +j] = sum;
            }
        }
        return result;
    }
}

// Matrix addition
impl ops::Sum<Matrix> for Matrix {
    fn sum(self, Matrix::second) -> Matrix {
        if self.cols != second.cols || self.rows != second.rows {
            panic!("Matrix dimensions do not match!");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i]+second.data[i];
        }
        return result;
    }
}

// Multiplication of Matrix by a scalar
impl ops::Mul<f64> for Matrix {
    fn mul(self, f64:scalar)-> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        if scalar == 0{
            return result;
        }
        for i in 0..self.data.len() {
            result.data[i] = self.data[i]*scalar;
        }
        return result;
    }
}