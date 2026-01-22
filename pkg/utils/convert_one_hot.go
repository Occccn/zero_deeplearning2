package utils

import (
	"gonum.org/v1/gonum/mat"
)

func ConvertOneHot(target *mat.Dense, vocab_size int) *mat.Dense {
	r, _ := target.Dims()
	one_hot := mat.NewDense(r, vocab_size, nil) // r × vocab_size
	for i := range r {
		idx := int(target.At(i, 0))
		one_hot.Set(i, idx, 1)
	}
	return one_hot
}
