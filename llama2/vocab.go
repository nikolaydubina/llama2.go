package llama2

import (
	"encoding/binary"
	"io"
	"log"
)

type Vocab struct {
	Words       []string
	Scores      []float32
	MaxTokenLen int // unused in Go version
}

func NewVocabFromFile(vocabSize int, r io.Reader) Vocab {
	vocab := Vocab{
		Words:  make([]string, 0, vocabSize),
		Scores: make([]float32, 0, vocabSize),
	}

	var maxTokenLen int32
	binary.Read(r, Endian, &maxTokenLen)
	vocab.MaxTokenLen = int(maxTokenLen)

	for i := 0; i < vocabSize; i++ {
		var score float32
		binary.Read(r, Endian, &score)
		vocab.Scores = append(vocab.Scores, score)

		var len int32
		binary.Read(r, Endian, &len)

		var word []byte = make([]byte, len)
		binary.Read(r, Endian, word)
		vocab.Words = append(vocab.Words, string(word))
	}

	return vocab
}

func (v Vocab) EncodeWord(s string) int {
	for i, word := range v.Words {
		if word == s {
			return i
		}
	}
	return -1
}

func (v Vocab) Encode(s string) (tokens []int) {
	// first encode every individual byte in the input string
	for i := 0; i < len(s); i++ {
		id := v.EncodeWord(string(s[i : i+1]))
		if id == -1 {
			log.Fatalf("bad token(%v)", string(s[i:i+1]))
		}
		tokens = append(tokens, id)
	}

	// merge the best consecutive pair each iteration, according the scores
	for len(tokens) > 1 {
		bestScore, bestID, bestIdx := float32(-1e10), -1, -1
		for i := 0; i < len(tokens)-1; i++ {
			// check if we can merge the pair (tokens[i], tokens[i+1])
			if id := v.EncodeWord(v.Words[tokens[i]] + v.Words[tokens[i+1]]); id != -1 && v.Scores[id] > bestScore {
				// this merge pair exists in vocab! record its score and position
				bestScore, bestID, bestIdx = v.Scores[id], id, i
			}
		}

		if bestIdx == -1 {
			// we couldn't find any more pairs to merge, so we are done
			break
		}

		// merge the consecutive pair (bestIdx, bestIdx+1) into new token bestID
		tokens[bestIdx] = bestID
		// remove the token at bestIdx+1, shift the entire sequence back 1
		copy(tokens[bestIdx+1:], tokens[bestIdx+2:])
		tokens = tokens[:len(tokens)-1]
	}

	return tokens
}
