package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestConvertOneHot(t *testing.T) {
	tests := []struct {
		name       string
		target     *mat.Dense
		vocab_size int
		expected   *mat.Dense
	}{
		{
			name:       "Test ConvertOneHot",
			target:     mat.NewDense(1, 1, []float64{0}),
			vocab_size: 1,
			expected:   mat.NewDense(1, 1, []float64{1}),
		},
		{
			name:       "Test ConvertOneHot with vocab_size=2",
			target:     mat.NewDense(2, 1, []float64{0, 1}), // 2要素に修正
			vocab_size: 2,
			expected: mat.NewDense(2, 2, []float64{
				1, 0,
				0, 1,
			}),
		},
		{
			name:       "Test ConvertOneHot with vocab_size=3",
			target:     mat.NewDense(3, 1, []float64{1, 2, 2}),
			vocab_size: 3,
			expected: mat.NewDense(3, 3, []float64{
				0, 1, 0,
				0, 0, 1,
				0, 0, 1,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertOneHot(tt.target, tt.vocab_size)
			if !mat.Equal(got, tt.expected) {
				t.Errorf("ConvertOneHot() = %v, want %v", got, tt.expected)
			}
		})
	}
}
