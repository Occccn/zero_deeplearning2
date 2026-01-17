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

// func TestPPMICalculation(t *testing.T) {
// 	tests := []struct {
// 		name      string
// 		co_matrix *mat.Dense
// 		expected  *mat.Dense
// 	}{
// 		{
// 			name: "Test PPMICalculation",
// 			co_matrix: mat.NewDense(3, 3, []float64{
// 				0, 1, 0,
// 				1, 0, 1,
// 				0, 1, 0,
// 			}),
// 			expected: mat.NewDense(3, 3, []float64{
// 				0, 1, 0,
// 				1, 0, 1,
// 				0, 1, 0,
// 			}),
// 		},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			got := PPMICalculation(tt.co_matrix)
// 			if !mat.Equal(got, tt.expected) {
// 				t.Errorf("PPMICalculation() = %v, want %v", got, tt.expected)
// 			}
// 		})
// 	}
// }
