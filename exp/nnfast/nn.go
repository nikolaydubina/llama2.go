package nnfast

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

var NumThreads = 8

func Acc[T float32 | float64](a, b []T) {
	for i := range a {
		a[i] += b[i]
	}
}

// RMSNorm is Root Mean Square Normalization
func RMSNorm[T float32 | float64](o, x, weight []T) {
	// calculate sum of squares
	var ss T
	for _, v := range x {
		ss += v * v
	}
	ss /= T(len(x))
	ss += 1e-5
	ss = T(math.Sqrt(float64(ss)))
	// normalize and scale
	for i := range o {
		o[i] = weight[i] * x[i] / ss
	}
}

func SoftMax[T float32 | float64](x []T) {
	// find max for numerical stability
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	// exp and sum
	var sum T
	for i := range x {
		x[i] = T(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	// normalize
	for i := range x {
		x[i] /= sum
	}
}

// MatMulUnroll4 in to multiple inlined operations.
// W (d,n) @ x (n,) -> xout (d,)
func MatMulUnroll4[T float32 | float64](xout, x, w []T) {
	for i := range xout {
		var sum T
		j := 0
		for ; (j + 4) < len(x); j += 4 {
			sum += w[i*len(x)+j] * x[j]
			sum += w[i*len(x)+j+1] * x[j+1]
			sum += w[i*len(x)+j+2] * x[j+2]
			sum += w[i*len(x)+j+3] * x[j+3]
		}
		for ; j < len(x); j++ {
			sum += w[i*len(x)+j] * x[j]
		}
		xout[i] = sum
	}
}

// MatMulParallel chunks horizontally across cache lines and parallelizes
func MatMulParallel[T float32 | float64](xout, x, w []T) {
	n, m := len(xout), len(x)
	if n < NumThreads {
		MatMulUnroll4(xout, x, w)
		return
	}
	var wg sync.WaitGroup
	wg.Add(NumThreads)
	for i := 0; i < NumThreads; i++ {
		rowStart := i * n / NumThreads
		rowEnd := (i + 1) * n / NumThreads
		if i == NumThreads-1 {
			rowEnd = n
		}
		go func(rowStart, rowEnd int) { MatMulUnroll4(xout[rowStart:rowEnd], x, w[m*rowStart:m*rowEnd]); wg.Done() }(rowStart, rowEnd)
	}
	wg.Wait()
}

// MatMul uses multiple optimizations
func MatMul[T float32 | float64](xout, x, w []T) { MatMulParallel(xout, x, w) }

func ArgMax[T float32 | float64](v []T) int {
	maxi, maxv := 0, v[0]
	for i, v := range v {
		if v > maxv {
			maxv, maxi = v, i
		}
	}
	return maxi
}

// Sample index from probabilities, they must sum to 1
func Sample[T float32 | float64](probabilities []T) int {
	r := T(rand.Float32())
	var cdf T
	for i, p := range probabilities {
		cdf += p
		if r < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}

// SampleTopP ("top-p" sampling, or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
// Notes on llama2.c: here not reusing probability index slice, since practically it is as fast to request new one.
func SampleTopP[T float32 | float64](probabilities []T, topp T) int {
	type PI struct {
		prob  T
		index int
	}
	pis := make([]PI, 0, len(probabilities))

	// quicksort indices in descending order of probabilities
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	// so for efficiency we crop these out as candidates before sorting
	cutoff := (1.0 - topp) / T(len(probabilities)-1)
	for i, p := range probabilities {
		if p >= cutoff {
			pis = append(pis, PI{prob: p, index: i})
		}
	}
	sort.Slice(pis, func(i, j int) bool { return pis[i].prob > pis[j].prob })

	// truncate the list where cumulative probability exceeds topp
	cumulativeProb := T(0)
	lastIdx := len(pis) - 1 // in case of rounding errors consider all elements
	for i, pi := range pis {
		cumulativeProb += pi.prob
		if cumulativeProb > topp {
			lastIdx = i
			break // we've exceeded topp by including lastIdx
		}
	}

	// sample from the truncated list
	r := T(rand.Float32()) * cumulativeProb
	cdf := T(0)
	for i := 0; i <= lastIdx; i++ {
		cdf += pis[i].prob
		if r < cdf {
			return pis[i].index
		}
	}

	return pis[lastIdx].index // in case of rounding errors
}
