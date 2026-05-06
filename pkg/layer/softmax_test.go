package layer

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmax(t *testing.T) {
	const tolerance = 1e-5
	tests := []struct {
		name     string
		x        *mat.Dense
		expected *mat.Dense
	}{
		{
			name:     "single sample",
			x:        mat.NewDense(1, 3, []float64{1, 2, 3}),
			expected: mat.NewDense(1, 3, []float64{0.09003057, 0.24472847, 0.66524096}),
		},
		{
			name: "batch size 2",
			x: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				1, 1, 1,
			}),
			expected: mat.NewDense(2, 3, []float64{
				0.09003057, 0.24472847, 0.66524096,
				0.33333333, 0.33333333, 0.33333333,
			}),
		},
		{
			name:     "numerically stable for large logits",
			x:        mat.NewDense(1, 3, []float64{1000, 1001, 1002}),
			expected: mat.NewDense(1, 3, []float64{0.09003057, 0.24472847, 0.66524096}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := Softmax(tt.x)
			rows, cols := output.Dims()
			expectedRows, expectedCols := tt.expected.Dims()

			if rows != expectedRows || cols != expectedCols {
				t.Fatalf("Softmax() dims = (%d, %d), want (%d, %d)", rows, cols, expectedRows, expectedCols)
			}

			for r := 0; r < rows; r++ {
				var rowSum float64
				for c := 0; c < cols; c++ {
					got := output.At(r, c)
					want := tt.expected.At(r, c)
					if math.Abs(want-got) > tolerance {
						t.Errorf("Softmax()[%d,%d] = %v, want %v", r, c, got, want)
					}
					rowSum += got
				}
				if math.Abs(1.0-rowSum) > tolerance {
					t.Errorf("Softmax() row sum at row %d = %v, want 1.0", r, rowSum)
				}
			}
		})
	}
}
