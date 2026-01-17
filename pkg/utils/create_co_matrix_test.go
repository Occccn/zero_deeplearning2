package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCreateCoMatrix(t *testing.T) {
	tests := []struct {
		name        string
		corpus      []int64
		vocab_size  int
		window_size int
		co_matrix   *mat.Dense
	}{
		{
			name:        "Test CreateCoMatrix",
			corpus:      []int64{0, 1, 2, 3, 4, 5},
			vocab_size:  6,
			window_size: 1,
			co_matrix: mat.NewDense(6, 6, []float64{
				0, 1, 0, 0, 0, 0,
				1, 0, 1, 0, 0, 0,
				0, 1, 0, 1, 0, 0,
				0, 0, 1, 0, 1, 0,
				0, 0, 0, 1, 0, 1,
				0, 0, 0, 0, 1, 0,
			}),
		},
		{
			name:        "test window_size=2",
			corpus:      []int64{0, 1, 2, 3, 4, 5},
			vocab_size:  6,
			window_size: 2,
			co_matrix: mat.NewDense(6, 6, []float64{
				0, 1, 1, 0, 0, 0,
				1, 0, 1, 1, 0, 0,
				1, 1, 0, 1, 1, 0,
				0, 1, 1, 0, 1, 1,
				0, 0, 1, 1, 0, 1,
				0, 0, 0, 1, 1, 0,
			}),
		},
		{
			name:        "test small corpus",
			corpus:      []int64{0, 1, 2},
			vocab_size:  3,
			window_size: 1,
			co_matrix: mat.NewDense(3, 3, []float64{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}),
		},
		{
			name:        "test single element",
			corpus:      []int64{0},
			vocab_size:  1,
			window_size: 1,
			co_matrix:   mat.NewDense(1, 1, []float64{0}),
		},
		{
			name:        "test repeated words",
			corpus:      []int64{0, 1, 0, 1}, // word 0 と 1 が交互
			vocab_size:  2,
			window_size: 1,
			co_matrix: mat.NewDense(2, 2, []float64{
				0, 1, // word 0 の隣は word 1
				1, 0, // word 1 の隣は word 0
			}),
		},
		{
			name:        "test window larger than corpus",
			corpus:      []int64{0, 1, 2},
			vocab_size:  3,
			window_size: 5, // corpus より大きい
			co_matrix: mat.NewDense(3, 3, []float64{
				0, 1, 1,
				1, 0, 1,
				1, 1, 0,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			co_matrix := CreateCoMatrix(tt.corpus, tt.vocab_size, tt.window_size)
			if !mat.Equal(co_matrix, tt.co_matrix) {
				t.Errorf("CreateCoMatrix() = %v, want %v", co_matrix, tt.co_matrix)
			}
		})
	}
}
