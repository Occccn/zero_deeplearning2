package layer

import (
	"gonum.org/v1/gonum/mat"
)

type SoftmaxWithLoss struct {
	Loss   float64
	Y      *mat.Dense
	Target *mat.Dense
}

func (softmax_with_loss *SoftmaxWithLoss) Forward(x *mat.Dense, target *mat.Dense) float64 {
	softmax_with_loss.Target = target

	softmax_with_loss.Y = Softmax(x)
	softmax_with_loss.Loss = CrossEntropyError(softmax_with_loss.Y, softmax_with_loss.Target)
	return softmax_with_loss.Loss
}

func (softmax_with_loss *SoftmaxWithLoss) Backward(dout float64) *mat.Dense {
	batchSize, _ := softmax_with_loss.Target.Dims()
	rows, cols := softmax_with_loss.Y.Dims()
	targetRows, targetCols := softmax_with_loss.Target.Dims()
	dx := mat.NewDense(rows, cols, nil)
	dx.Copy(softmax_with_loss.Y)

	if targetRows != rows {
		panic("target rows must match prediction rows")
	}

	// one-hot labels
	if targetCols == cols {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				dx.Set(i, j, dx.At(i, j)-softmax_with_loss.Target.At(i, j))
			}
		}
	} else {
		// index labels
		for i := 0; i < rows; i++ {
			targetIdx := int(softmax_with_loss.Target.At(i, 0))
			if targetIdx < 0 || targetIdx >= cols {
				panic("target index out of range")
			}
			dx.Set(i, targetIdx, dx.At(i, targetIdx)-1.0)
		}
	}

	scale := dout / float64(batchSize)
	dx.Scale(scale, dx)
	return dx
}
