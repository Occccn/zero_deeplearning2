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
		{
			name: "Test CrossEntropyError by batch size 2 with one-hot",
			y: mat.NewDense(2, 3, []float64{
				0.1, 0.2, 0.7, // サンプル1: 正解は2
				0.3, 0.6, 0.1, // サンプル2: 正解は1
			}),
			t: mat.NewDense(2, 3, []float64{
				0, 0, 1, // 正解ラベル: 2
				0, 1, 0, // 正解ラベル: 1
			}),
			// (-log(0.7 + 1e-7) + -log(0.6 + 1e-7)) / 2
			expected: (0.3566749 + 0.5108256) / 2, // ≈ 0.4337503
		},
		{
			name: "Test CrossEntropyError by batch size 2 with index label",
			y: mat.NewDense(2, 3, []float64{
				0.1, 0.2, 0.7,
				0.3, 0.6, 0.1,
			}),
			t:        mat.NewDense(2, 1, []float64{2, 1}), // インデックス形式
			expected: (0.3566749 + 0.5108256) / 2,
		},
		{
			name: "Test CrossEntropyError by batch size 3",
			y: mat.NewDense(3, 4, []float64{
				0.1, 0.05, 0.8, 0.05, // 正解は2
				0.1, 0.7, 0.1, 0.1, // 正解は1
				0.9, 0.05, 0.025, 0.025, // 正解は0
			}),
			t: mat.NewDense(3, 1, []float64{2, 1, 0}),
			// (-log(0.8+1e-7) + -log(0.7+1e-7) + -log(0.9+1e-7)) / 3
			expected: (0.2231435 + 0.3566749 + 0.1053605) / 3, // ≈ 0.2283930
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
