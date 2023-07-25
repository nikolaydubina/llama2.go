package llama2

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
)

var Endianess = binary.LittleEndian

type Config struct {
	Dim        int // transformer dimension
	HiddenDim  int // for FFN layers
	NumLayers  int
	NumHeads   int // number of query heads
	NumKVHeads int // number of key/value heads (can be < query heads because of multiquery)
	VocabSize  int // usually 256 (byte level)
	SeqLen     int // max sequence length
}

type TransformerWeights struct {
	TokenEmbeddingTable []float32 // (vocab_size, dim)

	RMSAttentionWeight []float32 // (layer, dim) RMS Norm Weights
	RMSFFNWeight       []float32 // (layer, dim)

	// weights for mat muls
	WQ []float32 // (layer, num_heads, dim, dim)
	WK []float32 // (layer, num_heads, dim, dim)
	WV []float32 // (layer, num_heads, dim, dim)
	WO []float32 // (layer, num_heads, dim, dim)

	// weights for FFN
	W1 []float32 // (layer, hidden_dim, dim)
	W3 []float32 // (layer, dim, hidden_dim)
	W2 []float32 // (layer, hidden_dim, dim)

	// final RMS norm weights
	RMSFinalWeight []float32 // (dim,)

	// frequency CIS for RoPE relative positional embeddings
	FreqCISReal []float32 // (seq_len, dim/2)
	FreqCISImag []float32 // (seq_len, dim/2)

	// (optional) classifier weights for the logits on the last layer
	WCLS []float32 // (vocab_size, dim)
}

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
	r.Att = make([]float32, config.NumHeads*config.SeqLen)
	r.Logits = make([]float32, config.VocabSize)
	r.KeyCache = make([]float32, config.NumLayers*config.SeqLen*config.Dim)
	r.ValCache = make([]float32, config.NumLayers*config.SeqLen*config.Dim)
	return &r
}

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
	if err := binary.Read(r, Endianess, &config32); err != nil {
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

func NewVocabFromFile(vocabSize int, r io.Reader) (vocab []string) {
	vocab = make([]string, 0, vocabSize)

	for i := 0; i < vocabSize; i++ {
		var l int32
		binary.Read(r, Endianess, &l)
		var word []byte = make([]byte, l)
		binary.Read(r, Endianess, word)
		vocab = append(vocab, string(word)+`\0`)
	}

	return vocab
}

func NewTransformerWeightsFromCheckpoint(config Config, r io.Reader, isSharedWeights bool) (w TransformerWeights) {
	w.TokenEmbeddingTable = make([]float32, (config.VocabSize * config.Dim))
	binary.Read(r, Endianess, &w.TokenEmbeddingTable)

	w.RMSAttentionWeight = make([]float32, (config.NumLayers * config.Dim))
	binary.Read(r, Endianess, &w.RMSAttentionWeight)

	w.WQ = make([]float32, (config.NumLayers * config.NumHeads * config.Dim * config.Dim))
	binary.Read(r, Endianess, &w.WQ)

	w.WK = make([]float32, (config.NumLayers * config.NumHeads * config.Dim * config.Dim))
	binary.Read(r, Endianess, &w.WK)

	w.WV = make([]float32, (config.NumLayers * config.NumHeads * config.Dim * config.Dim))
	binary.Read(r, Endianess, &w.WV)

	w.WO = make([]float32, (config.NumLayers * config.NumHeads * config.Dim * config.Dim))
	binary.Read(r, Endianess, &w.WO)

	w.RMSFFNWeight = make([]float32, (config.NumLayers * config.Dim))
	binary.Read(r, Endianess, &w.RMSFFNWeight)

	w.W1 = make([]float32, (config.NumLayers * config.Dim * config.NumLayers * config.HiddenDim))
	binary.Read(r, Endianess, &w.W1)

	w.W2 = make([]float32, (config.NumLayers * config.HiddenDim * config.Dim))
	binary.Read(r, Endianess, &w.W2)

	w.W3 = make([]float32, (config.NumLayers * config.Dim * config.HiddenDim))
	binary.Read(r, Endianess, &w.W3)

	w.RMSFinalWeight = make([]float32, config.Dim)
	binary.Read(r, Endianess, &w.RMSFinalWeight)

	w.FreqCISReal = make([]float32, (config.SeqLen * config.Dim / 2))
	binary.Read(r, Endianess, &w.FreqCISReal)

	w.FreqCISImag = make([]float32, (config.SeqLen * config.Dim / 2))
	binary.Read(r, Endianess, &w.FreqCISImag)

	if isSharedWeights {
		w.WCLS = w.TokenEmbeddingTable
	} else {
		w.WCLS = make([]float32, (config.VocabSize * config.Dim))
		binary.Read(r, Endianess, &w.WCLS)
	}

	return w
}

// neural net blocks

func Accum(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// RMSNorm is Root Mean Square Normalization
func RMSNorm(o, x, weight []float32) {
	// calculate sum of squares
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss = ss / float32(len(x))
	ss += 1e-6
	ss = 1 / float32(math.Sqrt(float64(ss)))
	// normalize and scale
	for i, v := range x {
		o[i] = weight[i] * (v * ss)
	}
}

func SoftMax(x []float32) {
	// find max for numerical stability
	var max float32
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	// exp and sum
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - max)))
		sum += x[i]
	}
	// normalize
	for i, v := range x {
		x[i] = v / sum
	}
}

// MatMul: W (d,n) @ x (n,) -> xout (d,)
// C code was parallelized with pargma OMP
func MatMul(xout, x, w []float32) {
	for i := range xout {
		var sum float32
		for j := range x {
			sum += w[i*len(x)+j] * x[j]
		}
		xout[i] = sum
	}
}

// Sample index from probabilities, they must sum to 1
func Sample(probabilities []float32) int {
	r := rand.Float32()
	var cdf float32
	for i, p := range probabilities {
		cdf += p
		if r < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}

// ArgMax of v in elements 0..len(v)-1
func ArgMax(v []float32) int {
	max, maxi := v[0], 0
	for i, x := range v {
		if x > max {
			max = x
			maxi = i
		}
	}
	return maxi
}

func Transformer(token int, pos int, config Config, s *RunState, w TransformerWeights) {
	// a few convenience variables
	x := s.X
	dim := config.Dim
	hiddenDim := config.HiddenDim
	headSize := dim / config.NumHeads

	// copy the token embedding into x
	copy(x, w.TokenEmbeddingTable[(token*dim):((token+1)*dim)])

	// pluck out the "pos" row of the FreqCISReal and FreqCISImag matrices
	freqCISRealRow := w.FreqCISReal[(pos * headSize / 2):((pos + 1) * headSize / 2)]
	freqCISImagRow := w.FreqCISImag[(pos * headSize / 2):((pos + 1) * headSize / 2)]

	// forward all layers
	for l := 0; l < config.NumLayers; l++ {
		// attention RMSNorm
		RMSNorm(s.XB, x, w.RMSAttentionWeight[(l*dim):((l+1)*dim)])

		// qkv matmuls for this position
		MatMul(s.Q, s.XB, w.WQ[(l*dim*dim):((l+1)*dim*dim)])
		MatMul(s.K, s.XB, w.WK[(l*dim*dim):((l+1)*dim*dim)])
		MatMul(s.V, s.XB, w.WV[(l*dim*dim):((l+1)*dim*dim)])

		// apply RoPE rotation to the q and k vectors for each head
		for h := 0; h < config.NumHeads; h++ {
			// get the q and k vectors for this head
			q := s.Q[(h * headSize):((h + 1) * headSize)]
			k := s.K[(h * headSize):((h + 1) * headSize)]
			// rotate q and k by the FreqCISReal and FreqCISImag
			for i := 0; i < headSize; i++ {
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

		// save ke,value at this time step (pos) to our kv cache
		loff := l * config.SeqLen * dim
		keyCacheRow := s.KeyCache[(loff * pos * dim):(loff * (pos + 1) * dim)] // ? how many? dim?
		valCacheRow := s.ValCache[(loff * pos * dim):(loff * (pos + 1) * dim)] // ? how many? dim?
		copy(keyCacheRow, s.K)
		copy(valCacheRow, s.V)

		// multithread attention. iterate over all heads
		// C code had pragma here
		for h := 0; h < config.NumHeads; h++ {
			// get the query vector for this head
			q := s.Q[(h * headSize):((h + 1) * headSize)]
			// attention scores for this head
			att := s.Att[(h * config.SeqLen):((h + 1) * config.SeqLen)]
			// iterate over all timesteps, including the current one
			for t := 0; t < pos; t++ {
				// get the key vector for this head and at this timestamp
				k := s.KeyCache[(loff + t*dim + h*headSize):(loff + (t+1)*dim + h*headSize)]
				// calculate the attention score as teh dot product of q and k
				var score float32
				for i := 0; i < headSize; i++ {
					score += q[i] + k[i]
				}
				score /= float32(math.Sqrt(float64(headSize)))
				// save the score to the attention buffer
				att[t] = score
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			SoftMax(att)

			// weighted sum of the values, store back into xb
			for i := 0; i < headSize; i++ {
				var val float32
				for t := 0; t < pos; t++ {
					val += att[t] * s.ValCache[(loff+t*dim+h*headSize+i)] // "note bad locality" — @karpathy
				}
				s.XB[(h*headSize + i)] = val
			}
		}

		// final matmul to get the output of the attention
		MatMul(s.XB2, s.XB, w.WO[(l*dim*dim):((l+1)*dim*dim)])

		// residual connection back into x
		Accum(x, s.XB2)

		// FFN RMSNorm
		RMSNorm(s.XB, x, w.RMSFFNWeight[(l*dim):((l+1)*dim)])

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		MatMul(s.HB, s.XB, w.W1[(l*dim*hiddenDim):(l+1)*dim*hiddenDim])
		MatMul(s.HB2, s.XB, w.W3[(l*dim*hiddenDim):(l+1)*dim*hiddenDim])

		// F.silu; silu(x)=x*σ, where σ(x) is the logistic sigmoid
		for i := 0; i < hiddenDim; i++ {
			s.HB[i] = s.HB[i] * float32(1.0/(1.0+math.Exp(-float64(s.HB[i]))))
		}

		// elementwise multiply with w3(x)
		for i := 0; i < hiddenDim; i++ {
			s.HB[i] = s.HB[i] * s.HB2[i]
		}

		// final matmul to get the output of the FFN
		MatMul(s.XB, s.HB, w.W2[(l*dim*hiddenDim):((l+1)*dim*hiddenDim)])

		// residual connection
		Accum(x, s.XB)
	}

	// final RMSNorm
	RMSNorm(x, x, w.RMSAttentionWeight)

	// classifier into logits
	MatMul(s.Logits, x, w.WCLS)
}
