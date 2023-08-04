package llama2

import (
	"encoding/binary"
	"io"
	"log"
	"strings"
)

const (
	BOS = "<s>\n"
)

// TokenDecoder converts tokens into strings to writer.
type TokenDecoder struct {
	BOS_ID          int
	tokenizer       Tokenizer
	out             io.StringWriter
	stripWhiteSpace bool
}

func NewTokenDecoder(tokenizer Tokenizer, out io.StringWriter) *TokenDecoder {
	return &TokenDecoder{
		BOS_ID:    1,
		tokenizer: tokenizer,
		out:       out,
	}
}

func (v *TokenDecoder) WriteToken(tokens ...int) {
	var b strings.Builder
	for _, token := range tokens {
		// following BOS token (1), sentencepiece decoder strips any leading whitespace
		if token == 1 {
			v.stripWhiteSpace = true
			b.WriteString(BOS)
			continue
		}
		if v.stripWhiteSpace && v.tokenizer.Words[token][0] == ' ' {
			v.stripWhiteSpace = false
			b.WriteString(v.tokenizer.Words[token][1:])
			continue
		}
		v.out.WriteString(v.tokenizer.Words[token])
	}
}

type Tokenizer struct {
	Words       []string
	Scores      []float32
	MaxTokenLen int // unused in Go version
	BOS_ID      int
	EOS_ID      int
}

func NewTokenizerFromFile(vocabSize int, r io.Reader) Tokenizer {
	tokenizer := Tokenizer{
		Words:  make([]string, 0, vocabSize),
		Scores: make([]float32, 0, vocabSize),
		BOS_ID: 1, // llama2 uses BOS = 1 in tokenizer
		EOS_ID: 2, // llama2 uses EOS = 2 in tokenizer
	}

	var maxTokenLen int32
	binary.Read(r, Endian, &maxTokenLen)
	tokenizer.MaxTokenLen = int(maxTokenLen)

	for i := 0; i < vocabSize; i++ {
		var score float32
		binary.Read(r, Endian, &score)
		tokenizer.Scores = append(tokenizer.Scores, score)

		var len int32
		binary.Read(r, Endian, &len)

		var word []byte = make([]byte, len)
		binary.Read(r, Endian, word)
		tokenizer.Words = append(tokenizer.Words, string(word))
	}

	return tokenizer
}

func (v Tokenizer) Decoder(out io.StringWriter) *TokenDecoder { return NewTokenDecoder(v, out) }

func (v Tokenizer) EncodeWord(s string) int {
	for i, word := range v.Words {
		if word == s {
			return i
		}
	}
	return -1
}

func (v Tokenizer) Encode(s string) (tokens []int) {
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
