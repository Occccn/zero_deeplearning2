package main

import (
	"fmt"

	"github.com/Occccn/zero_deeplearning2/pkg/utils"
	"gonum.org/v1/gonum/mat"
)

func main() {
	text := "You say goodbye and I say hello."
	corpus, _, _ := utils.Preprocess(text)
	co_matrix := utils.CreateCoMatrix(corpus, 7, 1)
	ppmi_matrix := utils.PPMICalculation(co_matrix)

	// SVD分解
	var svd mat.SVD
	ok := svd.Factorize(ppmi_matrix, mat.SVDFull)
	if !ok {
		panic("SVD failed")
	}

	// 結果を取得
	u := mat.Dense{}
	svd.UTo(&u) // 左特異ベクトル U
	fmt.Println("U:")
	fmt.Println(u)
	fmt.Println("co_matrix:")
	fmt.Println(co_matrix)
	fmt.Println("ppmi_matrix:")
	fmt.Println(ppmi_matrix)
}
