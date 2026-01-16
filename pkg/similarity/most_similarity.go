package similarity

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type similarityPair struct {
	index int64
	score float64
}

func MostSimilarity(query string, word_to_id map[string]int64, id_to_word map[int64]string, word_matrix *mat.Dense, top_k int) {
	_, ok := word_to_id[query]
	if !ok {
		println("word not found")
		return
	}
	for _, word := range id_to_word {
		if word == query {
			query_id := word_to_id[query]
			query_vector := word_matrix.RowView(int(query_id))

			vocab_size, _ := word_matrix.Dims()
			sims := make([]similarityPair, 0, vocab_size)
			for i := 0; i < vocab_size; i++ {
				sims = append(sims, similarityPair{
					index: int64(i),
					score: mat.Dot(query_vector, word_matrix.RowView(i)),
				})
			}
			count := 0

			// 類似度に基づいて降順ソート
			sort.Slice(sims, func(i, j int) bool {
				return sims[i].score > sims[j].score
			})
			for _, sim := range sims {
				if count >= top_k {
					break
				}
				if sim.index == query_id {
					continue
				}
				count++
				fmt.Print("word: ", id_to_word[sim.index], " similarity: ", sim.score)

			}
		}

	}
}
