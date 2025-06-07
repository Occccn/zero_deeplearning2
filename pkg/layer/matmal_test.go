package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMatmal(t *testing.T) {

	test := []struct {
		name   string
		x      *mat.Dense
		Params *mat.Dense
	}{
		{
			name: "Test Matmal Forward and Backward",
			x: mat.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			Params: mat.NewDense(3, 2, []float64{
				1, 2,
				3, 4,
				5, 6,
			}),
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			matmal := &Matmal{
				Params: tt.Params,
				Grad:   mat.NewDense(tt.Params.RawMatrix().Rows, tt.Params.RawMatrix().Cols, nil),
			}

			// Forward pass
			output := matmal.Forward(tt.x)
			expectedOutput := mat.NewDense(2, 2, []float64{
				22, 28,
				49, 64,
			})
			if !mat.Equal(output, expectedOutput) {
				t.Errorf("Forward output = %v, want %v", mat.Formatted(output), mat.Formatted(expectedOutput))
			}

			// Backward pass
			dout := mat.NewDense(2, 2, []float64{
				1, 1,
				1, 1,
			})
			dx := matmal.Backward(dout)
			expectedDx := mat.NewDense(2, 3, []float64{
				3, 7, 11,
				3, 7, 11,
			})
			if !mat.Equal(dx, expectedDx) {
				t.Errorf("Backward dx = %v, want %v", mat.Formatted(dx), mat.Formatted(expectedDx))
			}

			expectedGrad := mat.NewDense(3, 2, []float64{
				5, 5,
				7, 7,
				9, 9,
			})
			if !mat.Equal(matmal.Grad, expectedGrad) {
				t.Errorf("Backward grad = %v, want %v", mat.Formatted(matmal.Grad), mat.Formatted(expectedGrad))
			}
		})
	}
}
