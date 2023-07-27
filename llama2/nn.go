package llama2

import (
	"math"
	"math/rand"
)

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
	ss /= float32(len(x))
	ss += 1e-5
	ss = float32(math.Sqrt(float64(ss)))
	// normalize and scale
	for i := range o {
		o[i] = weight[i] * x[i] / ss
	}
}

func SoftMax(x []float32) {
	// find max for numerical stability
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	// exp and sum
	var sum float32
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	// normalize
	for i := range x {
		x[i] /= sum
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

func ArgMax(v []float32) int {
	maxi, maxv := 0, v[0]
	for i, v := range v {
		if v > maxv {
			maxv, maxi = v, i
		}
	}
	return maxi
}
