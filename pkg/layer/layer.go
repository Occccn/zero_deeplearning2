package layer

import (
	"gonum.org/v1/gonum/mat"
)

// interface: レイヤーの共通の振る舞いを定義
type Layer interface {
	Forward(x mat.Matrix) mat.Matrix
	Backward(dout mat.Matrix) mat.Matrix
}
