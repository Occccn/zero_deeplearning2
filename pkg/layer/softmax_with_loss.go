package layer

import (
	"gonum.org/v1/gonum/mat"
)

// class SoftmaxWithLoss:
//     def __init__(self):
//         self.params, self.grads = [], []
//         self.y = None  # softmaxの出力
//         self.t = None  # 教師ラベル

//     def forward(self, x, t):
//         self.t = t
//         self.y = softmax(x)

//         # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
//         if self.t.size == self.y.size:
//             self.t = self.t.argmax(axis=1)

//         loss = cross_entropy_error(self.y, self.t)
//         return loss

//     def backward(self, dout=1):
//         batch_size = self.t.shape[0]

//         dx = self.y.copy()
//         dx[np.arange(batch_size), self.t] -= 1
//         dx *= dout
//         dx = dx / batch_size

//         return dx

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
	batch_size, _ := softmax_with_loss.Target.Dims()
	rows, cols := softmax_with_loss.Y.Dims()
	dx := mat.NewDense(
		rows,
		cols,
		nil,
	)

	for i := range rows {
		for j := range cols {
			dx.Set(i, j, softmax_with_loss.Y.At(i, j)*dout/float64(batch_size))
			if softmax_with_loss.Target.At(i, j) == 1 {
				dx.Set(i, j, dx.At(i, j)-1)
			}
		}
	}
	return dx
}
