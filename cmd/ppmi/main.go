package main

import (
	"fmt"

	"github.com/Occccn/zero_deeplearning2/pkg/utils"
)

func main() {
	text := "You say goodbye and I say hello."
	corpus, _, _ := utils.Preprocess(text)
	co_matrix := utils.CreateCoMatrix(corpus, 7, 1)
	ppmi_matrix := utils.PPMICalculation(co_matrix)
	fmt.Println("co_matrix:")
	fmt.Println(co_matrix)
	fmt.Println("ppmi_matrix:")
	fmt.Println(ppmi_matrix)
}
