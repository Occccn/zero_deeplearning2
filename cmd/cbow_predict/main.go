package main

import (
	"fmt"

	"github.com/Occccn/zero_deeplearning2/pkg/layer"
	"gonum.org/v1/gonum/mat"
)

func main() {
	c0 := mat.NewDense(1, 7, []float64{1, 0, 0, 0, 0, 0, 0})
	c1 := mat.NewDense(1, 7, []float64{0, 0, 1, 0, 0, 0, 0})
	W_in := mat.NewDense(7, 3, []float64{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		0.1, 0.2, 0.3,
	})
	w_out := mat.NewDense(3, 7, []float64{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		0.1, 0.2, 0.3,
	})
	in_layer0 := layer.Matmal{
		Params: W_in,
		Grad:   mat.NewDense(7, 3, nil),
		X:      c0,
	}
	in_layer1 := layer.Matmal{
		Params: W_in,
		Grad:   mat.NewDense(7, 3, nil),
		X:      c1,
	}
	h0 := in_layer0.Forward(c0)
	h1 := in_layer1.Forward(c1)
	h := mat.NewDense(1, 3, nil)
	h.Add(h0, h1)
	h.Scale(0.5, h)
	out := layer.Matmal{
		Params: w_out,
		Grad:   mat.NewDense(3, 7, nil),
		X:      h,
	}
	y := out.Forward(h)
	fmt.Println(y)
}
