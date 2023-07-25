package llama2

import (
	"encoding/binary"
	"io"
)

func NewVocabFromFile(vocabSize int, r io.Reader) (vocab []string) {
	vocab = make([]string, 0, vocabSize)

	for i := 0; i < vocabSize; i++ {
		var l int32
		binary.Read(r, Endian, &l)
		var word []byte = make([]byte, l)
		binary.Read(r, Endian, word)
		vocab = append(vocab, string(word))
	}

	return vocab
}