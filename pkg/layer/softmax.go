package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Softmax(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	if r != 1 {
		panic("Softmax only supports a 1-row matrix")
	}
	var output *mat.Dense = mat.NewDense(r, c, nil)
	var max_value float64 = x.At(0, 0)
	var sum_exp_value float64
	for i := range c {
		v := x.At(0, i)
		if v > max_value {
			max_value = v
		}
	}
	for i := range c {
		sum_exp_value += math.Exp(x.At(0, i) - max_value)
	}
	for i := range c {
		v := math.Exp(x.At(0, i)-max_value) / sum_exp_value
		output.Set(0, i, v)
	}
	return output
}
