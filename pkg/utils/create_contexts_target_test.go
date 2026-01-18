package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCreateContextTarget(t *testing.T) {
	tests := []struct {
		name        string
		corpus      []int64
		window_size int
		expected    *mat.Dense
	}{
		{
			name:        "Test CreateContextTarget",
			corpus:      []int64{0, 1, 2, 3, 4, 5},
			window_size: 1,
			expected: mat.NewDense(4, 2, []float64{
				0, 2,
				1, 3,
				2, 4,
				3, 5,
			})},
		{
			name:        "Test CreateContextTarget with window_size=2",
			corpus:      []int64{0, 1, 2, 3, 4, 5},
			window_size: 2,
			expected: mat.NewDense(2, 4, []float64{
				0, 1, 3, 4,
				1, 2, 4, 5,
			})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CreateContextsTarget(tt.corpus, tt.window_size)
			if !mat.Equal(got, tt.expected) {
				t.Errorf("CreateContextTarget() = %v, want %v", got, tt.expected)
			}
		})
	}
}
