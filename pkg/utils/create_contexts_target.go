package utils

import (
	"gonum.org/v1/gonum/mat"
)

func CreateContextsTarget(corpus []int64, window_size int) *mat.Dense {
	if len(corpus)-2 < 0 {
		panic("corpus is too short")
	}
	contexts := mat.NewDense(len(corpus)-2, window_size*2, nil)
	for i := range len(corpus) - 2 {
		colIdx := 0 // 列インデックスを別でカウント
		for j := -window_size; j <= window_size; j++ {
			if j == 0 {
				continue
			}
			idx := i + j + 1
			contexts.Set(i, colIdx, float64(corpus[idx]))
			colIdx++
		}
	}
	return contexts
}
