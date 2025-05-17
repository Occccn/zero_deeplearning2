package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	a := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	b := mat.NewDense(2, 2, []float64{5, 6, 7, 8})

	var c mat.Dense
	c.Mul(a, b)

	fmt.Printf("Result:\n%v\n", mat.Formatted(&c))
}
