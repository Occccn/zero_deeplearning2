package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Softmax(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	output := mat.NewDense(rows, cols, nil)

	for r := 0; r < rows; r++ {
		maxValue := x.At(r, 0)
		for c := 0; c < cols; c++ {
			v := x.At(r, c)
			if v > maxValue {
				maxValue = v
			}
		}

		var sumExpValue float64
		for c := 0; c < cols; c++ {
			sumExpValue += math.Exp(x.At(r, c) - maxValue)
		}

		for c := 0; c < cols; c++ {
			v := math.Exp(x.At(r, c)-maxValue) / sumExpValue
			output.Set(r, c, v)
		}
	}

	return output
}
