package layer

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmaxWithLossForward(t *testing.T) {
	const tolerance = 1e-3
	tests := []struct {
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
		{
			name: "Test SoftmaxWithLoss by batch size 2 with one-hot",
			x: mat.NewDense(2, 3, []float64{
				0, 0, 0,
				0, 0, 0,
			}),
			target: mat.NewDense(2, 3, []float64{
				1, 0, 0,
				0, 0, 1,
			}),
			expected: 1.098612,
		},
		{
			name: "Test SoftmaxWithLoss by batch size 2 with index label",
			x: mat.NewDense(2, 3, []float64{
				0, 0, 0,
				0, 0, 0,
			}),
			target:   mat.NewDense(2, 1, []float64{0, 2}),
			expected: 1.098612,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			softmax_with_loss := &SoftmaxWithLoss{}
			loss := softmax_with_loss.Forward(tt.x, tt.target)
			if math.Abs(loss-tt.expected) > tolerance {
				t.Errorf("SoftmaxWithLoss() = %v, want %v", loss, tt.expected)
			}
		})
	}
}

func TestSoftmaxWithLossBackward(t *testing.T) {
	const tolerance = 1e-5

	tests := []struct {
		name     string
		x        *mat.Dense
		target   *mat.Dense
		dout     float64
		expected *mat.Dense
	}{
		{
			name:   "backward with one-hot label",
			x:      mat.NewDense(1, 3, []float64{1, 2, 3}),
			target: mat.NewDense(1, 3, []float64{0, 0, 1}),
			dout:   1.0,
			expected: mat.NewDense(1, 3, []float64{
				0.09003057, 0.24472847, -0.33475904,
			}),
		},
		{
			name:   "backward with index label",
			x:      mat.NewDense(1, 3, []float64{1, 2, 3}),
			target: mat.NewDense(1, 1, []float64{2}),
			dout:   1.0,
			expected: mat.NewDense(1, 3, []float64{
				0.09003057, 0.24472847, -0.33475904,
			}),
		},
		{
			name: "backward by batch size 2",
			x: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				1, 1, 1,
			}),
			target: mat.NewDense(2, 3, []float64{
				0, 0, 1,
				1, 0, 0,
			}),
			dout: 1.0,
			expected: mat.NewDense(2, 3, []float64{
				0.04501529, 0.12236424, -0.16737952,
				-0.33333333, 0.16666667, 0.16666667,
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := &SoftmaxWithLoss{}
			layer.Forward(tt.x, tt.target)
			got := layer.Backward(tt.dout)

			rows, cols := got.Dims()
			expectedRows, expectedCols := tt.expected.Dims()
			if rows != expectedRows || cols != expectedCols {
				t.Fatalf("Backward dims = (%d, %d), want (%d, %d)", rows, cols, expectedRows, expectedCols)
			}

			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					if math.Abs(got.At(r, c)-tt.expected.At(r, c)) > tolerance {
						t.Errorf("Backward[%d,%d] = %v, want %v", r, c, got.At(r, c), tt.expected.At(r, c))
					}
				}
			}
		})
	}
}
