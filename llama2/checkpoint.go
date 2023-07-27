package llama2

import (
	"encoding/binary"
	"io"
)

var Endian = binary.LittleEndian

func NewConfigFromCheckpoint(r io.Reader) (Config, error) {
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
		return Config{}, err
	}
	config := Config{
		Dim:        int(config32.Dim),
		HiddenDim:  int(config32.HiddenDim),
		NumLayers:  int(config32.NumLayers),
		NumHeads:   int(config32.NumHeads),
		NumKVHeads: int(config32.NumKVHeads),
		VocabSize:  int(config32.VocabSize),
		SeqLen:     int(config32.SeqLen),
	}
	return config, nil
}

func NewTransformerWeightsFromCheckpoint(config Config, r io.Reader, isSharedWeights bool) TransformerWeights {
	w := TransformerWeights{
		TokenEmbeddingTable: make([]float32, (config.VocabSize * config.Dim)),
		RMSAttentionWeight:  make([]float32, (config.NumLayers * config.Dim)),
		RMSFFNWeight:        make([]float32, (config.NumLayers * config.Dim)),
		RMSFinalWeight:      make([]float32, config.Dim),
		WQ:                  make([]float32, (config.NumLayers * config.Dim * config.Dim)),
		WK:                  make([]float32, (config.NumLayers * config.Dim * config.Dim)),
		WV:                  make([]float32, (config.NumLayers * config.Dim * config.Dim)),
		WO:                  make([]float32, (config.NumLayers * config.Dim * config.Dim)),
		W1:                  make([]float32, (config.NumLayers * config.Dim * config.HiddenDim)),
		W2:                  make([]float32, (config.NumLayers * config.HiddenDim * config.Dim)),
		W3:                  make([]float32, (config.NumLayers * config.Dim * config.HiddenDim)),
		FreqCISReal:         make([]float32, (config.SeqLen * config.HeadSize() / 2)),
		FreqCISImag:         make([]float32, (config.SeqLen * config.HeadSize() / 2)),
		WCLS:                make([]float32, (config.VocabSize * config.Dim)),
	}

	binary.Read(r, Endian, w.TokenEmbeddingTable)
	binary.Read(r, Endian, w.RMSAttentionWeight)
	binary.Read(r, Endian, w.WQ)
	binary.Read(r, Endian, w.WK)
	binary.Read(r, Endian, w.WV)
	binary.Read(r, Endian, w.WO)
	binary.Read(r, Endian, w.RMSFFNWeight)
	binary.Read(r, Endian, w.W1)
	binary.Read(r, Endian, w.W2)
	binary.Read(r, Endian, w.W3)
	binary.Read(r, Endian, w.RMSFinalWeight)
	binary.Read(r, Endian, w.FreqCISReal)
	binary.Read(r, Endian, w.FreqCISImag)

	if isSharedWeights {
		w.WCLS = w.TokenEmbeddingTable
	} else {
		binary.Read(r, Endian, w.WCLS)
	}

	return w
}
