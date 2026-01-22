package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCreateContextTarget(t *testing.T) {
	tests := []struct {
		name        string
		corpus      []int64
		window_size int
		contexts    *mat.Dense
		target      *mat.Dense
	}{
		{
			name:        "Test CreateContextTarget",
			corpus:      []int64{0, 1, 2, 3, 4, 5},
			window_size: 1,
			contexts: mat.NewDense(4, 2, []float64{
				0, 2,
				1, 3,
				2, 4,
				3, 5,
			}),
			target: mat.NewDense(4, 1, []float64{
				1,
				2,
				3,
				4,
			}),
		},
		{
			name:        "Test CreateContextTarget with window_size=2",
			corpus:      []int64{0, 1, 2, 3, 4, 5},
			window_size: 2,
			contexts: mat.NewDense(2, 4, []float64{
				0, 1, 3, 4,
				1, 2, 4, 5,
			}),
			target: mat.NewDense(2, 1, []float64{
				2,
				3,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got_contexts, got_target := CreateContextsTarget(tt.corpus, tt.window_size)
			if !mat.Equal(got_contexts, tt.contexts) {
				t.Errorf("CreateContextTarget() = %v, want %v", got_contexts, tt.contexts)
			}
			if !mat.Equal(got_target, tt.target) {
				t.Errorf("CreateContextTarget() = %v, want %v", got_target, tt.target)
			}
		})
	}
}
