package utils

import "strings"

func Preprocess(text string) string {
	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, ".", " .")
	return text
}
