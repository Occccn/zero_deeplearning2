package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SumColumn(x *mat.Dense) *mat.Dense {
	_, col := x.Dims()
	S := mat.NewDense(1, col, nil)
	for i := range col {
		sum := mat.Sum(x.ColView(i))
		S.Set(0, i, sum)
	}
	return S
}

func PPMICalculation(co_matrix *mat.Dense) *mat.Dense {
	eps := 1e-8
	row, col := co_matrix.Dims()
	matrix := mat.NewDense(row, col, nil)
	N := mat.Sum(co_matrix)
	S := SumColumn(co_matrix)
	for i := range row {
		for j := range col {
			pmi := math.Log2(co_matrix.At(i, j) * N / (S.At(0, i)*S.At(0, j) + eps))
			matrix.Set(i, j, math.Max(pmi, 0))
		}
	}
	return matrix
}
