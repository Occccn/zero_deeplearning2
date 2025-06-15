package layer

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmax(t *testing.T) {
	const tolerance = 1e-5
	expected := mat.NewDense(1, 3, []float64{0.09003057, 0.24472847, 0.66524096})
	test := []struct {
		name string
		x    *mat.Dense
	}{
		{
			name: "Test Normal",
			x: mat.NewDense(1, 3, []float64{
				1, 2, 3,
			}),
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			output := Softmax(tt.x)

			if math.Abs(expected.At(0, 1)-output.At(0, 1)) > tolerance {
				t.Errorf("Softmax() = %v, want %v", expected.At(0, 1), output.At(0, 1))
			}
		})
	}

}
