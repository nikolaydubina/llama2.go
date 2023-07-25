package llama2

type RunState struct {
	// current wave of activations
	X      []float32 // (dim,) activation at current time stamp
	XB     []float32 // (dim,) same, but inside ar residual branch
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

func NewRunState(config Config) *RunState {
	var r RunState
	r.X = make([]float32, config.Dim)
	r.XB = make([]float32, config.Dim)
	r.XB2 = make([]float32, config.Dim)
	r.HB = make([]float32, config.HiddenDim)
	r.HB2 = make([]float32, config.HiddenDim)
	r.Q = make([]float32, config.Dim)
	r.K = make([]float32, config.Dim)
	r.V = make([]float32, config.Dim)
	r.Att = make([]float32, (config.NumHeads * config.SeqLen))
	r.Logits = make([]float32, config.VocabSize)
	r.KeyCache = make([]float32, (config.NumLayers * config.SeqLen * config.Dim))
	r.ValCache = make([]float32, (config.NumLayers * config.SeqLen * config.Dim))
	return &r
}
