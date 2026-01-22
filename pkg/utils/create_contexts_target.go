package utils

import (
	"gonum.org/v1/gonum/mat"
)

func CreateContextsTarget(corpus []int64, window_size int) (*mat.Dense, *mat.Dense) {
	rowCount := len(corpus) - 2*window_size
	if rowCount < 1 {
		panic("corpus is too short")
	}
	contexts := mat.NewDense(rowCount, window_size*2, nil)
	target := mat.NewDense(rowCount, 1, nil)
	for i := range rowCount {
		target.Set(i, 0, float64(corpus[i+window_size]))
		colIdx := 0
		targetIdx := i + window_size // target の位置
		for j := -window_size; j <= window_size; j++ {
			if j == 0 {
				continue
			}
			contexts.Set(i, colIdx, float64(corpus[targetIdx+j]))
			colIdx++
		}
	}
	return contexts, target
}
