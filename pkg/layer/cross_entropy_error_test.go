package layer

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropyError(t *testing.T) {
	const tolerance = 1e-6

	tests := []struct {
		name     string
		y        *mat.Dense
		t        *mat.Dense
		expected float64
	}{
		{
			name:     "Test CrossEntropyError by batch size 1",
			y:        mat.NewDense(1, 3, []float64{0.1, 0.2, 0.7}),
			t:        mat.NewDense(1, 3, []float64{0, 0, 1}),
			expected: 0.3566749,
		},
		{
			name:     "Test CrossEntropyError with index label",
			y:        mat.NewDense(1, 3, []float64{0.1, 0.2, 0.7}),
			t:        mat.NewDense(1, 1, []float64{2}), // インデックス形式
			expected: 0.3566749,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CrossEntropyError(tt.y, tt.t)
			if math.Abs(got-tt.expected) > tolerance {
				t.Errorf("CrossEntropyError() = %v, want %v (tolerance: %v)", got, tt.expected, tolerance)
			}
		})
	}
}
