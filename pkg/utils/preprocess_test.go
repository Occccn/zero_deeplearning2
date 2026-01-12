package utils

import (
	"reflect"
	"testing"
)

func TestPreprocess(t *testing.T) {
	tests := []struct {
		name       string
		text       string
		corpus     []int64 // ← フィールド名を corpus に
		word_to_id map[string]int64
		id_to_word map[int64]string
	}{
		{
			name:       "Test Preprocess",
			text:       "Hello, world.",
			corpus:     []int64{0, 1, 2},
			word_to_id: map[string]int64{"hello,": 0, "world": 1, ".": 2},
			id_to_word: map[int64]string{0: "hello,", 1: "world", 2: "."},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corpus, word_to_id, id_to_word := Preprocess(tt.text)
			if !reflect.DeepEqual(corpus, tt.corpus) {
				t.Errorf("Preprocess() = %v, corpus: %v", corpus, tt.corpus)
			}
			if !reflect.DeepEqual(word_to_id, tt.word_to_id) {
				t.Errorf("Preprocess() = %v, word_to_id: %v", word_to_id, tt.word_to_id)
			}
			if !reflect.DeepEqual(id_to_word, tt.id_to_word) {
				t.Errorf("Preprocess() = %v, id_to_word: %v", id_to_word, tt.id_to_word)
			}
		})
	}
}
