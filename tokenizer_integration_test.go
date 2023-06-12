package tokenizers

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type embeddingsSample struct {
	Text            string          `json:"text"`
	Tokens          []string        `json:"tokens"`
	TokenizerOutput tokenizerOutput `json:"tokenizer_output"`
	Embeddings      [][]float32     `json:"embeddings"`
}

type tokenizerOutput struct {
	InputIDs      []uint32 `json:"input_ids"`
	AttentionMask []uint32 `json:"attention_mask"`
}

func TestEncodeAdequacy(t *testing.T) {
	tk, err := FromFile("./fixtures/tokenizer.json")
	require.NoError(t, err)
	defer tk.Close()

	readFile, err := os.Open("./fixtures/test_embeddings.jsonl")
	require.NoError(t, err)

	fileScanner := bufio.NewScanner(readFile)
	fileScanner.Split(bufio.ScanLines)

	for fileScanner.Scan() {
		data := fileScanner.Bytes()
		sample := &embeddingsSample{}
		err := json.Unmarshal(data, sample)
		require.NoError(t, err)

		actual := tk.Encode(sample.Text, true)
		assert.Equal(t, sample.TokenizerOutput.InputIDs[:len(actual)], actual)
	}

	readFile.Close()
}
