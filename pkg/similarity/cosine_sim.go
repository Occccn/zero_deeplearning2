package similarity

import (
	"gonum.org/v1/gonum/mat"
)

func CosineSimilarity(a, b mat.Vector) float64 {
	aNorm := mat.Norm(a, 2)
	bNorm := mat.Norm(b, 2)
	dotProduct := mat.Dot(a, b)
	cosineSim := dotProduct / (aNorm * bNorm)
	return cosineSim
}
