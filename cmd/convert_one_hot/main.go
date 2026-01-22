package main

import (
	"github.com/Occccn/zero_deeplearning2/pkg/utils"
)

func main() {
	text := "You say goodbye and I say hello."
	corpus, _, _ := utils.Preprocess(text)
	contexts, target := utils.CreateContextsTarget(corpus, 1)
}
