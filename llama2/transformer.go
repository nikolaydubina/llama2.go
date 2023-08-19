package llama2

import (
	"math"
	"sync"

	nn "github.com/nikolaydubina/llama2.go/exp/nnfast"
)

type RunState struct {
	// current wave of activations

	X      []float32 // (dim,) activation at current time stamp
	XB     []float32 // (dim,) same, but inside a residual branch
	XB2    []float32 // (dim,) an additional buffer just for convenience
	HB     []float32 // (hidden_dim,) buffer for hidden dimension in the FFN
	HB2    []float32 // (hidden_dim,) buffer for hidden dimension in the FFN
	Q      []float32 // (dim,) query
	K      []float32 // (kv_dim,) key
	V      []float32 // (kv_dim,) value
	Att    []float32 // (n_heads, seq_len) buffer for scores/attention values
	Logits []float32 // (vocab_size) output logits

	// kv cache

	KCache []float32 // (layer, seq_len, kv_dim)
	VCache []float32 // (layer, seq_len, kv_dim)
}

func NewRunState(config Config) RunState {
	return RunState{
		X:      make([]float32, config.Dim),
		XB:     make([]float32, config.Dim),
		XB2:    make([]float32, config.Dim),
		HB:     make([]float32, config.HiddenDim),
		HB2:    make([]float32, config.HiddenDim),
		Q:      make([]float32, config.Dim),
		K:      make([]float32, config.KVDim()),
		V:      make([]float32, config.KVDim()),
		Att:    make([]float32, (config.NumHeads * config.SeqLen)),
		Logits: make([]float32, config.VocabSize),
		KCache: make([]float32, (config.NumLayers * config.SeqLen * config.KVDim())),
		VCache: make([]float32, (config.NumLayers * config.SeqLen * config.KVDim())),
	}
}

type TransformerWeights struct {
	TokenEmbeddingTable []float32 // (vocab_size, dim)

	RMSAttentionWeight []float32 // (num_layers, dim)
	RMSFFNWeight       []float32 // (num_layers, dim)
	RMSFinalWeight     []float32 // (dim,)

	// weights for mat muls
	// dim == n_heads * head_size

	WQ []float32 // (num_layers, dim, n_heads * head_size)
	WK []float32 // (num_layers, dim, n_kv_heads * head_size)
	WV []float32 // (num_layers, dim, n_kv_heads * head_size)
	WO []float32 // (num_layers, n_heads * head_size, dim)

	// weights for FFN

	W1 []float32 // (num_layers, dim, hidden_dim)
	W2 []float32 // (num_layers, hidden_dim, dim)
	W3 []float32 // (num_layers, dim, hidden_dim)

	// Deprecated: frequency CIS for RoPE relative positional embeddings

	FreqCISReal []float32 // (seq_len, head_size / 2) Deprecated
	FreqCISImag []float32 // (seq_len, head_size / 2) Deprecated

	// (optional) classifier weights for the logits on the last layer

	WCLS []float32 // (vocab_size, dim)
}

func Transformer(token int, pos int, config Config, s RunState, w TransformerWeights) {
	var wg sync.WaitGroup

	// a few convenience variables
	x := s.X
	dim := config.Dim
	kvDim := config.KVDim()
	kvMul := config.KVMul()
	hiddenDim := config.HiddenDim
	headSize := config.HeadSize()

	// copy the token embedding into x
	copy(x, w.TokenEmbeddingTable[token*dim:(token+1)*dim])

	// forward all layers
	for l := 0; l < config.NumLayers; l++ {
		// attention RMSNorm
		nn.RMSNorm(s.XB, x, w.RMSAttentionWeight[l*dim:((l+1)*dim)])

		// Q,K,V matmuls for this position
		wg.Add(3)
		go func() { nn.MatMul(s.Q, s.XB, w.WQ[l*dim*dim:(l+1)*dim*dim]); wg.Done() }()
		go func() { nn.MatMul(s.K, s.XB, w.WK[l*dim*kvDim:(l+1)*dim*kvDim]); wg.Done() }()
		go func() { nn.MatMul(s.V, s.XB, w.WV[l*dim*kvDim:(l+1)*dim*kvDim]); wg.Done() }()
		wg.Wait()

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		for i := 0; i+1 < dim; i += 2 {
			headDim := i % headSize
			freq := 1.0 / math.Pow(10000, float64(headDim)/float64(headSize))
			val := float64(pos) * freq
			fcr := float32(math.Cos(val))
			fci := float32(math.Sin(val))

			// how many vectors? 2 = q & k, 1 = q only
			rotN := 1
			if i < kvDim {
				rotN = 2
			}

			for v := 0; v < rotN; v++ {
				vec := s.K
				if v == 0 {
					vec = s.Q
				}
				v0, v1 := vec[i], vec[i+1]
				vec[i] = v0*fcr - v1*fci
				vec[i+1] = v0*fci + v1*fcr
			}
		}

		// save key and val at this time step (pos) to cache
		loff := l * config.SeqLen * kvDim
		copy(s.KCache[(loff+pos*kvDim):(loff+(pos+1)*kvDim)], s.K)
		copy(s.VCache[(loff+pos*kvDim):(loff+(pos+1)*kvDim)], s.V)

		// multihead attention. iterate over all heads
		// Notes on llama2.c: pragma here, using goroutines
		wg.Add(config.NumHeads)
		for h := 0; h < config.NumHeads; h++ {
			go func(h int) {
				defer wg.Done()

				// get the query vector for this head
				q := s.Q[(h * headSize):((h + 1) * headSize)]
				// attention scores for this head
				att := s.Att[(h * config.SeqLen):((h + 1) * config.SeqLen)]
				// iterate over all timesteps, including the current one
				for t := 0; t <= pos; t++ {
					// get the key vector for this head and at this timestamp
					k := s.KCache[(loff + t*kvDim + (h/kvMul)*headSize):(loff + t*kvDim + ((h+1)/kvMul)*headSize)]
					// calculate the attention score as the dot product of q and k
					var score float32
					for i := 0; i < headSize; i++ {
						score += q[i] * k[i]
					}
					score /= float32(math.Sqrt(float64(headSize)))
					// save the score to the attention buffer
					att[t] = score
				}

				// scores to get attention weights, from 0..pos inclusively
				nn.SoftMax(att[:pos+1])

				// weighted sum of the values, store back into xb
				clear(s.XB[(h * headSize):((h + 1) * headSize)])
				for t := 0; t <= pos; t++ {
					a := att[t]
					for i := 0; i < headSize; i++ {
						s.XB[((h * headSize) + i)] += a * s.VCache[loff+t*kvDim+(h/kvMul)*headSize+i]
					}
				}
			}(h)
		}
		wg.Wait()

		// final matmul to get the output of the attention
		nn.MatMul(s.XB2, s.XB, w.WO[l*dim*dim:(l+1)*dim*dim])

		// residual connection back into x
		nn.Acc(x, s.XB2)

		// FFN RMSNorm
		nn.RMSNorm(s.XB, x, w.RMSFFNWeight[l*dim:(l+1)*dim])

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		wg.Add(2)
		go func() { nn.MatMul(s.HB, s.XB, w.W1[l*dim*hiddenDim:(l+1)*dim*hiddenDim]); wg.Done() }()
		go func() { nn.MatMul(s.HB2, s.XB, w.W3[l*dim*hiddenDim:(l+1)*dim*hiddenDim]); wg.Done() }()
		wg.Wait()

		// F.silu; silu(x)=x*σ, where σ(x) is the logistic sigmoid
		for i := 0; i < hiddenDim; i++ {
			s.HB[i] /= (1.0 + float32(math.Exp(-float64(s.HB[i]))))
		}

		// elementwise multiply with w3(x)
		for i := 0; i < hiddenDim; i++ {
			s.HB[i] *= s.HB2[i]
		}

		// final matmul to get the output of the FFN
		nn.MatMul(s.XB, s.HB, w.W2[l*dim*hiddenDim:(l+1)*dim*hiddenDim])

		// residual connection
		nn.Acc(x, s.XB)
	}

	// final RMSNorm
	nn.RMSNorm(x, x, w.RMSFinalWeight)

	// classifier into logits
	nn.MatMul(s.Logits, x, w.WCLS)
}
