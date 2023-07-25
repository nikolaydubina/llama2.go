package llama2

type RunState struct {
	// current wave of activations
	X      []float32 // (dim,) activation at current time stamp
	XB     []float32 // (dim,) same, but inside a residual branch
	XB2    []float32 // (dim,) an additional buffer just for convenience
	HB     []float32 // (hidden_dim,) buffer for hidden dimension in the FFN
	HB2    []float32 // (hidden_dim,) buffer for hidden dimension in the FFN
	Q      []float32 // (dim,) query
	K      []float32 // (dim,) key
	V      []float32 // (dim,) value
	Att    []float32 // (n_heads, seq_len) buffer for scores/attention values
	Logits []float32 // output logits
	// kv cache
	KeyCache []float32 // (layer, seq_len, dim)
	ValCache []float32 // (layer, seq_len, dim)
}

func NewRunState(config Config) RunState {
	return RunState{
		X:        make([]float32, config.Dim),
		XB:       make([]float32, config.Dim),
		XB2:      make([]float32, config.Dim),
		HB:       make([]float32, config.HiddenDim),
		HB2:      make([]float32, config.HiddenDim),
		Q:        make([]float32, config.Dim),
		K:        make([]float32, config.Dim),
		V:        make([]float32, config.Dim),
		Att:      make([]float32, (config.NumHeads * config.SeqLen)),
		Logits:   make([]float32, config.VocabSize),
		KeyCache: make([]float32, (config.NumLayers * config.SeqLen * config.Dim)),
		ValCache: make([]float32, (config.NumLayers * config.SeqLen * config.Dim)),
	}
}
