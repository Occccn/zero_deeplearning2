package utils

import (
	"testing"
)

func TestPreprocess(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "Test Preprocess",
			text: "Hello, world",
			want: "hello, world",
		},
		{
			name: "Test Preprocess",
			text: "Hello, world.",
			want: "hello, world .",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Preprocess(tt.text)
			if got != tt.want {
				t.Errorf("Preprocess() = %v, want %v", got, tt.want)
			}
		})
	}
}
