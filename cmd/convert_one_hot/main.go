package main

import (
	"fmt"

	"github.com/Occccn/zero_deeplearning2/pkg/utils"
)

func main() {
	text := "You say goodbye and I say hello."
	corpus, word_to_id, _ := utils.Preprocess(text)
	contexts, target := utils.CreateContextsTarget(corpus, 1)
	fmt.Println(target)
	target_one_hot := utils.ConvertOneHot(target, len(word_to_id))
	contexts_one_hot := utils.ConvertOneHot(contexts, len(word_to_id))
	fmt.Println(target_one_hot)
	fmt.Println(contexts_one_hot)
}
