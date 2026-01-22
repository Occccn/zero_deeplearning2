package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func CrossEntropyError(y *mat.Dense, t *mat.Dense) float64 {
	yRows, yCols := y.Dims()
	tRows, tCols := t.Dims()

	if yRows != 1 {
		panic("CrossEntropyError only supports a 1-row matrix")
	}

	var targetIdx int

	// 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
	if tRows == yRows && tCols == yCols {
		// one-hot形式: argmaxで変換
		for i := range yCols {
			if t.At(0, i) == 1 {
				targetIdx = i
				break
			}
		}
	} else {
		// インデックス形式: そのまま使用
		targetIdx = int(t.At(0, 0))
	}

	return -math.Log(y.At(0, targetIdx) + 1e-7)
}
