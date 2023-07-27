package llama2

import (
	"math"
)

type TransformerWeights struct {
	TokenEmbeddingTable []float32

	RMSAttentionWeight []float32
	RMSFFNWeight       []float32
	RMSFinalWeight     []float32

	// weights for mat muls
	WQ []float32
	WK []float32
	WV []float32
	WO []float32

	// weights for FFN
	W1 []float32
	W2 []float32
	W3 []float32

	// frequency CIS for RoPE relative positional embeddings
	FreqCISReal []float32
	FreqCISImag []float32

	// (optional) classifier weights for the logits on the last layer
	WCLS []float32
}

func Transformer(token int, pos int, config Config, s RunState, w TransformerWeights) {
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
		RMSNorm(s.XB, x, w.RMSAttentionWeight[l*dim:((l+1)*dim)])

		// qkv matmuls for this position
		MatMul(s.Q, s.XB, w.WQ[l*dim*dim:(l+1)*dim*dim])
		MatMul(s.K, s.XB, w.WK[l*dim*dim:(l+1)*dim*dim])
		MatMul(s.V, s.XB, w.WV[l*dim*dim:(l+1)*dim*dim])

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
		keyCacheRow := s.KeyCache[(loff + pos*dim):(loff + (pos+1)*dim)]
		valCacheRow := s.ValCache[(loff + pos*dim):(loff + (pos+1)*dim)]
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
			for t := 0; t <= pos; t++ {
				// get the key vector for this head and at this timestamp
				k := s.KeyCache[(loff + t*dim + h*headSize):(loff + (t+1)*dim + h*headSize)]
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
			SoftMax(att[:pos+1])

			// weighted sum of the values, store back into xb
			for i := 0; i < headSize; i++ {
				var val float32
				for t := 0; t <= pos; t++ {
					val += att[t] * s.ValCache[loff+t*dim+h*headSize+i] // "note bad locality" — @karpathy
				}
				s.XB[(h*headSize + i)] = val
			}
		}

		// final matmul to get the output of the attention
		MatMul(s.XB2, s.XB, w.WO[l*dim*dim:(l+1)*dim*dim])

		// residual connection back into x
		Accum(x, s.XB2)

		// FFN RMSNorm
		RMSNorm(s.XB, x, w.RMSFFNWeight[l*dim:(l+1)*dim])

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		MatMul(s.HB, s.XB, w.W1[l*dim*hiddenDim:(l+1)*dim*hiddenDim])
		MatMul(s.HB2, s.XB, w.W3[l*dim*hiddenDim:(l+1)*dim*hiddenDim])

		// F.silu; silu(x)=x*σ, where σ(x) is the logistic sigmoid
		for i := 0; i < hiddenDim; i++ {
			s.HB[i] /= (1.0 + float32(math.Exp(-float64(s.HB[i]))))
		}

		// elementwise multiply with w3(x)
		for i := 0; i < hiddenDim; i++ {
			s.HB[i] *= s.HB2[i]
		}

		// final matmul to get the output of the FFN
		MatMul(s.XB, s.HB, w.W2[l*dim*hiddenDim:(l+1)*dim*hiddenDim])

		// residual connection
		Accum(x, s.XB)
	}

	// final RMSNorm
	RMSNorm(x, x, w.RMSFinalWeight)

	// classifier into logits
	MatMul(s.Logits, x, w.WCLS)
}
