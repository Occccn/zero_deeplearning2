package main

import (
	"github.com/Occccn/zero_deeplearning2/pkg/similarity"
	"github.com/Occccn/zero_deeplearning2/pkg/utils"
)

func main() {
	text := "You say goodbye and I say hello."
	corpus, word_to_id, id_to_word := utils.Preprocess(text)
	co_matrix := utils.CreateCoMatrix(corpus, 1)
	similarity.MostSimilarity("you", word_to_id, id_to_word, co_matrix, 5)
}
