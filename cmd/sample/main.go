package main

import (
	"fmt"

	"github.com/Occccn/zero_deeplearning2/pkg/pjt2"
	"gonum.org/v1/gonum/mat"
)

func main() {
	a := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	b := mat.NewVecDense(4, []float64{2, 4, 6, 8})
	cosineSim := pjt2.CosineSimilarity(a, b)
	fmt.Printf("Cosine Similarity: %v\n", cosineSim)
}
