package utils

import (
	"gonum.org/v1/gonum/mat"
)

func CreateCoMatrix(corpus []int64) *mat.Dense {
	length := len(corpus)
	co_matrix := mat.NewDense(length, length, nil)
	for i, word_id1 := range corpus {
		left_index := i - 1
		right_index := i + 1
		if left_index >= 0 {
			left_word_id := corpus[left_index]
			co_matrix.Set(int(word_id1), int(left_word_id), 1)
		}
		if right_index < length {
			right_word_id := corpus[right_index]
			co_matrix.Set(int(word_id1), int(right_word_id), 1)
		}
	}
	return co_matrix
}
