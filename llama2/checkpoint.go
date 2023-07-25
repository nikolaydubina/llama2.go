package llama2

import (
	"encoding/binary"
	"fmt"
	"io"
)

var Endian = binary.LittleEndian

func NewConfigFromCheckpoint(r io.Reader) Config {
	// binary reader expects exact binary size for int
	var config32 struct {
		Dim        int32
		HiddenDim  int32
		NumLayers  int32
		NumHeads   int32
		NumKVHeads int32
		VocabSize  int32
		SeqLen     int32
	}
	if err := binary.Read(r, Endian, &config32); err != nil {
		fmt.Println("binary.Read failed:", err)
	}
	return Config{
		Dim:        int(config32.Dim),
		HiddenDim:  int(config32.HiddenDim),
		NumLayers:  int(config32.NumLayers),
		NumHeads:   int(config32.NumHeads),
		NumKVHeads: int(config32.NumKVHeads),
		VocabSize:  int(config32.VocabSize),
		SeqLen:     int(config32.SeqLen),
	}
}

func NewTransformerWeightsFromCheckpoint(config Config, r io.Reader, isSharedWeights bool) (w TransformerWeights) {
	w.TokenEmbeddingTable = make([]float32, (config.VocabSize * config.Dim))
	binary.Read(r, Endian, &w.TokenEmbeddingTable)

	w.RMSAttentionWeight = make([]float32, (config.NumLayers * config.Dim))
	binary.Read(r, Endian, &w.RMSAttentionWeight)

	w.WQ = make([]float32, (config.NumLayers * config.Dim * config.Dim))
	binary.Read(r, Endian, &w.WQ)

	w.WK = make([]float32, (config.NumLayers * config.Dim * config.Dim))
	binary.Read(r, Endian, &w.WK)

	w.WV = make([]float32, (config.NumLayers * config.Dim * config.Dim))
	binary.Read(r, Endian, &w.WV)

	w.WO = make([]float32, (config.NumLayers * config.Dim * config.Dim))
	binary.Read(r, Endian, &w.WO)

	w.RMSFFNWeight = make([]float32, (config.NumLayers * config.Dim))
	binary.Read(r, Endian, &w.RMSFFNWeight)

	w.W1 = make([]float32, (config.NumLayers * config.Dim * config.HiddenDim))
	binary.Read(r, Endian, &w.W1)

	w.W2 = make([]float32, (config.NumLayers * config.HiddenDim * config.Dim))
	binary.Read(r, Endian, &w.W2)

	w.W3 = make([]float32, (config.NumLayers * config.Dim * config.HiddenDim))
	binary.Read(r, Endian, &w.W3)

	w.RMSFinalWeight = make([]float32, config.Dim)
	binary.Read(r, Endian, &w.RMSFinalWeight)

	headSize := config.Dim / config.NumHeads

	w.FreqCISReal = make([]float32, (config.SeqLen * headSize / 2))
	binary.Read(r, Endian, &w.FreqCISReal)

	w.FreqCISImag = make([]float32, (config.SeqLen * headSize / 2))
	binary.Read(r, Endian, &w.FreqCISImag)

	if isSharedWeights {
		w.WCLS = w.TokenEmbeddingTable
	} else {
		w.WCLS = make([]float32, (config.VocabSize * config.Dim))
		binary.Read(r, Endian, &w.WCLS)
	}

	return w
}
