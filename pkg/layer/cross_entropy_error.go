package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func CrossEntropyError(y *mat.Dense, t *mat.Dense) float64 {
	batchSize, yCols := y.Dims()
	tRows, tCols := t.Dims()

	var totalLoss float64

	for b := range batchSize {
		var targetIdx int

		// one-hot形式
		if tRows == batchSize && tCols == yCols {
			for i := range yCols {
				if t.At(b, i) == 1 {
					targetIdx = i
					break
				}
			}
		} else {
			// インデックス形式
			targetIdx = int(t.At(b, 0))
		}

		totalLoss += -math.Log(y.At(b, targetIdx) + 1e-7)
	}

	return totalLoss / float64(batchSize) // 平均を返す
}
