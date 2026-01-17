package utils

import (
	"gonum.org/v1/gonum/mat"
)

func SumColumn(x *mat.Dense) *mat.Dense {
	_, col := x.Dims()
	S := mat.NewDense(1, col, nil)
	for i := 0; i < col; i++ {
		sum := mat.Sum(x.ColView(i))
		S.Set(0, i, sum)
	}
	return S
}

// func PPMICalculation(co_matrix *mat.Dense, verbose bool) *mat.Dense {
// 	eps := 1e-8
// 	row, col := co_matrix.Dims()
// 	matrix := mat.NewDense(row, col, nil)
// 	N := mat.Sum(co_matrix)
// 	for i := 0; i < row; i++ {

// 	rows, cols := co_matrix.Dims()
// 	for i := 0; i < rows; i++ {
// 		for j := 0; j < cols; j++ {
// 			co_matrix.Set(i, j, math.Log(co_matrix.At(i, j) / (rows * cols)))
// 		}
// 	}
// 	return co_matrix
// }
