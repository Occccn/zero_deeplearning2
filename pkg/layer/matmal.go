package layer

import (
	"gonum.org/v1/gonum/mat"
)

type Matmal struct {
	Params *mat.Dense
	Grad   *mat.Dense
	X      *mat.Dense
}

func (matmal *Matmal) Forward(x *mat.Dense) *mat.Dense {
	// np.dot(x, matmal.Params)
	output := mat.NewDense(x.RawMatrix().Rows, matmal.Params.RawMatrix().Cols, nil)
	output.Mul(x, matmal.Params)
	// matmal.X = x
	matmal.X = mat.NewDense(x.RawMatrix().Rows, x.RawMatrix().Cols, nil)
	matmal.X.Copy(x)
	return output
}

func (matmal *Matmal) Backward(dout *mat.Dense) *mat.Dense {
	// np.dot(grad, matmal.Params.T)
	dx := mat.NewDense(dout.RawMatrix().Rows, matmal.X.RawMatrix().Cols, nil)
	dx.Mul(dout, matmal.Params.T())

	// matmal.Grad = np.dot(matmal.X.T, grad)
	matmal.Grad = mat.NewDense(matmal.X.RawMatrix().Cols, dout.RawMatrix().Cols, nil)
	matmal.Grad.Mul(matmal.X.T(), dout)

	return dx
}
