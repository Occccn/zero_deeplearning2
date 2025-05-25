package similarity

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMostSimilarity(t *testing.T) {
	// テスト用の語彙とベクトル
	word_to_id := map[string]int64{
		"apple":  0,
		"banana": 1,
		"cherry": 2,
	}
	id_to_word := map[int64]string{
		0: "apple",
		1: "banana",
		2: "cherry",
	}

	word_matrix := []mat.Vector{
		mat.NewVecDense(2, []float64{1, 0}),   // apple
		mat.NewVecDense(2, []float64{0, 1}),   // banana
		mat.NewVecDense(2, []float64{1.1, 0}), // cherry
	}

	// 標準出力を一時的にパイプに切り替え
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// 実行
	MostSimilarity("apple", word_to_id, id_to_word, word_matrix, 1)

	// 出力を取得して復元
	_ = w.Close()
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	os.Stdout = oldStdout

	output := buf.String()
	fmt.Print(output)
	// 検証
	if !strings.Contains(output, "word: cherry") {
		t.Errorf("unexpected output:\n%s", output)
	}
}
