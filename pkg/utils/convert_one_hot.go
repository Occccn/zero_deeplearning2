package utils

import (
	"gonum.org/v1/gonum/mat"
)

func ConvertOneHot(target *mat.Dense, vocab_size int) *mat.Dense {
	r, c := target.Dims()
	if r != 1 {
		panic("target must be a 1-row matrix")
	}
	one_hot := mat.NewDense(vocab_size, c, nil)
	for i := range c {
		for j := range vocab_size {
			if j == int(target.At(0, i)) {
				one_hot.Set(j, i, 1)
			} else {
				one_hot.Set(j, i, 0)
			}
		}
	}
	return one_hot
}
