package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSumColumn(t *testing.T) {
	tests := []struct {
		name     string
		x        *mat.Dense
		expected *mat.Dense
	}{
		{
			name: "Test SumColumn",
			x: mat.NewDense(3, 3, []float64{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}),
			expected: mat.NewDense(1, 3, []float64{1, 2, 1}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SumColumn(tt.x)
			if !mat.Equal(got, tt.expected) {
				t.Errorf("SumColumn() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPPMICalculation(t *testing.T) {
	tests := []struct {
		name      string
		co_matrix *mat.Dense
		expected  *mat.Dense
	}{
		{
			name: "Test_2x2_diagonal",
			co_matrix: mat.NewDense(2, 2, []float64{
				2, 0,
				0, 2,
			}),
			// N = 4, S = [2, 2]
			// PPMI(0,0) = log2(2 * 4 / (2*2)) = log2(2) = 1
			expected: mat.NewDense(2, 2, []float64{
				1, 0,
				0, 1,
			}),
		},
		{
			name: "Test_uniform",
			co_matrix: mat.NewDense(2, 2, []float64{
				1, 1,
				1, 1,
			}),
			// N = 4, S = [2, 2]
			// PPMI(i,j) = log2(1 * 4 / (2*2)) = log2(1) = 0
			expected: mat.NewDense(2, 2, []float64{
				0, 0,
				0, 0,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PPMICalculation(tt.co_matrix)
			if !mat.EqualApprox(got, tt.expected, 1e-6) {
				t.Errorf("PPMICalculation() = %v, want %v", got, tt.expected)
			}
		})
	}
}
