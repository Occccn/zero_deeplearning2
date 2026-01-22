package utils

import (
	"gonum.org/v1/gonum/mat"
)

func ConvertOneHot(target *mat.Dense, vocab_size int) *mat.Dense {
	_, c := target.Dims()
	one_hot := mat.NewDense(c, vocab_size, nil) // c × vocab_size
	for i := range c {
		idx := int(target.At(0, i))
		one_hot.Set(i, idx, 1)
	}
	return one_hot
}
