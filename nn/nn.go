package nn

import (
	"math"
	"math/rand"
)

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

// MatMul: W (d,n) @ x (n,) -> xout (d,)
// C code was parallelized with pargma OMP
func MatMul[T float32 | float64](xout, x, w []T) {
	for i := range xout {
		var sum T
		for j := range x {
			sum += w[i*len(x)+j] * x[j]
		}
		xout[i] = sum
	}
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

func ArgMax[T float32 | float64](v []T) int {
	maxi, maxv := 0, v[0]
	for i, v := range v {
		if v > maxv {
			maxv, maxi = v, i
		}
	}
	return maxi
}
