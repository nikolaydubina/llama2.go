package llama2

type Config struct {
	Dim        int // transformer dimension
	HiddenDim  int // for FFN layers
	NumLayers  int
	NumHeads   int // number of query heads
	NumKVHeads int // number of key/value heads (can be < query heads because of multiquery)
	VocabSize  int // usually 256 (byte level)
	SeqLen     int // max sequence length
}

func (c Config) HeadSize() int { return c.Dim / c.NumHeads }

func (c Config) KVDim() int { return (c.Dim * c.NumKVHeads) / c.NumHeads }

// KVMul integer multiplier of the kv sharing in multiquery
func (c Config) KVMul() int { return c.NumHeads / c.NumKVHeads }
