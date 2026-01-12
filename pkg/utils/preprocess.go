package utils

import "strings"

func Preprocess(text string) ([]int64, map[string]int64, map[int64]string) {
	word_to_id := make(map[string]int64)
	id_to_word := make(map[int64]string)

	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, ".", " .")
	words := strings.Split(text, " ")
	for _, word := range words {
		if _, ok := word_to_id[word]; !ok {
			word_to_id[word] = int64(len(word_to_id))
			id_to_word[int64(len(id_to_word))] = word
		}
	}
	corpus := make([]int64, 0, len(words))
	for _, word := range words {
		corpus = append(corpus, word_to_id[word])
	}
	return corpus, word_to_id, id_to_word
}
