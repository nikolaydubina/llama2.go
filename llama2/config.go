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
