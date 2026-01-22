package layer

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmaxWithLoss(t *testing.T) {
	const tolerance = 1e-3
	test := []struct {
		name     string
		x        *mat.Dense
		target   *mat.Dense
		expected float64
	}{
		{
			name:     "Test SoftmaxWithLoss",
			x:        mat.NewDense(1, 3, []float64{1, 2, 3}),
			target:   mat.NewDense(1, 3, []float64{0, 0, 1}),
			expected: 0.4076,
		},
		{
			name:     "Test SoftmaxWithLoss with index label",
			x:        mat.NewDense(1, 3, []float64{1, 2, 3}),
			target:   mat.NewDense(1, 1, []float64{2}),
			expected: 0.4076,
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			softmax_with_loss := &SoftmaxWithLoss{}
			loss := softmax_with_loss.Forward(tt.x, tt.target)
			if math.Abs(loss-tt.expected) > tolerance {
				t.Errorf("SoftmaxWithLoss() = %v, want %v", loss, tt.expected)
			}
		})
	}
}
