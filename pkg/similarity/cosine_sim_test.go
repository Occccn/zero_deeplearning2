package similarity

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        *mat.VecDense
		b        *mat.VecDense
		expected float64
	}{
		{
			name:     "Same direction",
			a:        mat.NewVecDense(3, []float64{1, 2, 3}),
			b:        mat.NewVecDense(3, []float64{2, 4, 6}),
			expected: 1.0, // 同じ方向 → 類似度1
		},
		{
			name:     "Orthogonal vectors",
			a:        mat.NewVecDense(2, []float64{1, 0}),
			b:        mat.NewVecDense(2, []float64{0, 1}),
			expected: 0.0, // 直交 → 類似度0
		},
		{
			name:     "Opposite direction",
			a:        mat.NewVecDense(2, []float64{1, 0}),
			b:        mat.NewVecDense(2, []float64{-1, 0}),
			expected: -1.0, // 真逆 → 類似度-1
		},
	}

	const tolerance = 1e-4

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CosineSimilarity(tt.a, tt.b)
			if math.Abs(got-tt.expected) > tolerance {
				t.Errorf("CosineSimilarity() = %v, want %v", got, tt.expected)
			}
		})
	}
}
