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

type TransformerWeights struct {
	TokenEmbeddingTable []float32 // (vocab_size, dim)

	RMSAttentionWeight []float32 // (num_layers, dim)
	RMSFFNWeight       []float32 // (num_layers, dim)
	RMSFinalWeight     []float32 // (dim,)

	// weights for mat muls
	WQ []float32 // (num_layers, dim, dim)
	WK []float32 // (num_layers, dim, dim)
	WV []float32 // (num_layers, dim, dim)
	WO []float32 // (num_layers, dim, dim)

	// weights for FFN
	W1 []float32 // (num_layers, dim, hidden_dim)
	W2 []float32 // (num_layers, hidden_dim, dim)
	W3 []float32 // (num_layers, dim, hidden_dim)

	// frequency CIS for RoPE relative positional embeddings
	FreqCISReal []float32 // (seq_len, head_size / 2)
	FreqCISImag []float32 // (seq_len, head_size / 2)

	// (optional) classifier weights for the logits on the last layer
	WCLS []float32 // (vocab_size, dim)
}

// TransformerForward pass updates current run state and as outcome gets next token probabilities
func TransformerForward(token int, pos int, config Config, s RunState, w TransformerWeights) {
	var wg sync.WaitGroup

	// a few convenience variables
	x := s.X
	dim := config.Dim
	hiddenDim := config.HiddenDim
	headSize := config.HeadSize()

	// copy the token embedding into x
	copy(x, w.TokenEmbeddingTable[token*dim:(token+1)*dim])

	// pluck out the "pos" row of the FreqCISReal and FreqCISImag matrices
	freqCISRealRow := w.FreqCISReal[(pos * headSize / 2):((pos + 1) * headSize / 2)]
	freqCISImagRow := w.FreqCISImag[(pos * headSize / 2):((pos + 1) * headSize / 2)]

	// forward all layers
	for l := 0; l < config.NumLayers; l++ {
		// attention RMSNorm
		nn.RMSNorm(s.XB, x, w.RMSAttentionWeight[l*dim:((l+1)*dim)])

		// qkv matmuls for this position
		wg.Add(3)
		go func() { nn.MatMul(s.Q, s.XB, w.WQ[l*dim*dim:(l+1)*dim*dim]); wg.Done() }()
		go func() { nn.MatMul(s.K, s.XB, w.WK[l*dim*dim:(l+1)*dim*dim]); wg.Done() }()
		go func() { nn.MatMul(s.V, s.XB, w.WV[l*dim*dim:(l+1)*dim*dim]); wg.Done() }()
		wg.Wait()

		// apply RoPE rotation to the q and k vectors for each head
		for h := 0; h < config.NumHeads; h++ {
			// get the q and k vectors for this head
			q := s.Q[(h * headSize):((h + 1) * headSize)]
			k := s.K[(h * headSize):((h + 1) * headSize)]
			// rotate q and k by the FreqCISReal and FreqCISImag
			for i := 0; i < headSize; i += 2 {
				q0, q1 := q[i], q[i+1]
				k0, k1 := k[i], k[i+1]
				fcr := freqCISRealRow[i/2]
				fci := freqCISImagRow[i/2]
				q[i] = q0*fcr - q1*fci
				q[i+1] = q0*fci + q1*fcr
				k[i] = k0*fcr - k1*fci
				k[i+1] = k0*fci + k1*fcr
			}
		}

		// save key,value at this time step (pos) to our kv cache
		loff := l * config.SeqLen * dim
		copy(s.KeyCache[(loff+pos*dim):(loff+(pos+1)*dim)], s.K)
		copy(s.ValCache[(loff+pos*dim):(loff+(pos+1)*dim)], s.V)

		// multithread attention. iterate over all heads
		// C code had pragma here, using goroutines
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
					k := s.KeyCache[(loff + t*dim + h*headSize):(loff + t*dim + (h+1)*headSize)]
					// calculate the attention score as the dot product of q and k
					var score float32
					for i := 0; i < headSize; i++ {
						score += q[i] * k[i]
					}
					score /= float32(math.Sqrt(float64(headSize)))
					// save the score to the attention buffer
					att[t] = score
				}

				// softmax the scores to get attention weights, from 0..pos inclusively
				nn.SoftMax(att[:pos+1])

				// weighted sum of the values, store back into xb
				// llama2.c uses memset. resetting to zero in loop is ok since it is next iterated over same slice anyways.
				for i := 0; i < headSize; i++ {
					s.XB[(h*headSize + i)] = 0
				}
				for t := 0; t <= pos; t++ {
					a := att[t]
					for i := 0; i < headSize; i++ {
						s.XB[((h * headSize) + i)] += a * s.ValCache[loff+t*dim+h*headSize+i]
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
