package utils

import (
	"gonum.org/v1/gonum/mat"
)

func CreateCoMatrix(corpus []int64, window_size int) *mat.Dense {
	length := len(corpus)
	// ユニークな単語IDを数える
	unique := make(map[int64]bool)
	for _, id := range corpus {
		unique[id] = true
	}
	vocab_size := len(unique)
	co_matrix := mat.NewDense(vocab_size, vocab_size, nil)
	for i, word_id1 := range corpus {
		for j := 1; j <= window_size; j++ {
			left_index := i - j
			right_index := i + j
			if left_index >= 0 {
				left_word_id := corpus[left_index]
				co_matrix.Set(int(word_id1), int(left_word_id), 1)
			}
			if right_index < length {
				right_word_id := corpus[right_index]
				co_matrix.Set(int(word_id1), int(right_word_id), 1)
			}
		}
	}
	return co_matrix
}
