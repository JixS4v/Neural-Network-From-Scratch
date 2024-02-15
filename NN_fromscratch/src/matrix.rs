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

impl ops::Mul<Matrix> for Matrix {
    fn mul(self, Matrix:second) -> Result<Matrix> {
        let mut result = Matrix::new(self.rows, second.cols);
        if self.cols != second.rows {
            return Err("Matrix dimensions do not match");
        }
        for i in 0..self.rows {
            for j in 0..second.cols {
                let mut sum:f64 = 0;
                for k in 0..second.rows {
                    sum += self.data[i*self.cols +k] *  second.data[k*second.cols + j];
                }
                result.data[i*second.cols +j] = sum;
            }
        }
        return Ok(result);
    }
}