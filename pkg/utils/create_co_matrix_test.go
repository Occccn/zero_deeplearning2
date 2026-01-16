package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCreateCoMatrix(t *testing.T) {
	tests := []struct {
		name      string
		corpus    []int64
		co_matrix *mat.Dense
	}{
		{
			name:   "Test CreateCoMatrix",
			corpus: []int64{0, 1, 2, 3, 4, 5},
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
			name:   "test window_size=2",
			corpus: []int64{0, 1, 2, 3, 4, 5},
			co_matrix: mat.NewDense(6, 6, []float64{
				0, 1, 1, 0, 0, 0,
				1, 0, 1, 1, 0, 0,
				1, 1, 0, 1, 1, 0,
				0, 1, 1, 0, 1, 1,
				0, 0, 1, 1, 0, 1,
				0, 0, 0, 1, 1, 0,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			co_matrix := CreateCoMatrix(tt.corpus)
			if !mat.Equal(co_matrix, tt.co_matrix) {
				t.Errorf("CreateCoMatrix() = %v, want %v", co_matrix, tt.co_matrix)
			}
		})
	}
}
