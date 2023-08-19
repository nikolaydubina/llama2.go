package nnfast_test

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/nikolaydubina/llama2.go/exp/nnfast"
	"github.com/nikolaydubina/llama2.go/nn"
)

func TestAcc(t *testing.T) {
	a := []float32{1, 2, 3, 0, -1}
	b := []float32{4, 5, 6, 0, 1}
	nnfast.Acc(a, b)
	if a[0] != 5 || a[1] != 7 || a[2] != 9 || a[3] != 0 || a[4] != 0 {
		t.Errorf("Acc failed")
	}
}

func TestSoftMax(t *testing.T) {
	tests := []struct {
		x   []float32
		exp []float32
	}{
		{
			x:   []float32{1, 1, 2},
			exp: []float32{0.21194156, 0.21194156, 0.57611686},
		},
		{
			x:   []float32{0.5, -1, 12},
			exp: []float32{1.0129968e-05, 2.2603015e-06, 0.9999876},
		},
		{
			x:   []float32{0.2, 7, 13},
			exp: []float32{2.7539384e-06, 0.0024726165, 0.9975247},
		},
	}
	for i, tc := range tests {
		t.Run(fmt.Sprintf("%d: %#v", i, tc), func(t *testing.T) {
			nnfast.SoftMax(tc.x)
			if !slices.Equal(tc.exp, tc.x) {
				t.Errorf("got %v, exp %v", tc.x, tc.exp)
			}
		})
	}
}

func TestArgMax(t *testing.T) {
	tests := []struct {
		x   []float32
		exp int
	}{
		{
			x:   []float32{1, 1, 2},
			exp: 2,
		},
		{
			x:   []float32{0.5, -1, 12, 0},
			exp: 2,
		},
		{
			x:   []float32{0.2, 7, 13},
			exp: 2,
		},
		{
			x:   []float32{15, 7, 13},
			exp: 0,
		},
	}
	for i, tc := range tests {
		t.Run(fmt.Sprintf("%d: %#v", i, tc), func(t *testing.T) {
			if got := nnfast.ArgMax(tc.x); got != tc.exp {
				t.Errorf("got %d, exp %d", got, tc.exp)
			}
		})
	}
}

func TestMatMul(t *testing.T) {
	tests := []struct {
		x          []float32
		w          []float32
		exp        []float32
		numThreads int
	}{
		{
			x:          []float32{1, 2, 3, 4, 5},
			w:          []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			exp:        []float32{1 + 4 + 9 + 16 + 25, 6 + 14 + 24 + 36 + 50},
			numThreads: 8,
		},
		{
			x:          []float32{1, 2, 3},
			w:          []float32{1, 2, 3, 4, 5, 6},
			exp:        []float32{1 + 4 + 9, 4 + 10 + 18},
			numThreads: 8,
		},
		{
			x:          []float32{1, 2, 3},
			w:          []float32{1, 2, 3, 4, 5, 6},
			exp:        []float32{1 + 4 + 9, 4 + 10 + 18},
			numThreads: 2,
		},
		{
			x:          []float32{1, 2, 3},
			w:          []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			exp:        []float32{1 + 4 + 9, 4 + 10 + 18, 7 + 16 + 27},
			numThreads: 2,
		},
		{
			x:          []float32{1, 2, 3},
			w:          []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			exp:        []float32{1 + 4 + 9, 4 + 10 + 18, 7 + 16 + 27},
			numThreads: 3,
		},
	}
	for i, tc := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			got := make([]float32, len(tc.exp))
			nnfast.NumThreads = tc.numThreads
			nnfast.MatMul(got, tc.x, tc.w)
			if !slices.Equal(tc.exp, got) {
				t.Errorf("got %v, exp %v", got, tc.exp)
			}
		})
	}
}

func fillRand(x []float32, rnd *rand.Rand) {
	for i := range x {
		x[i] = rnd.Float32()
	}
}

func FuzzMatMul(f *testing.F) {
	f.Fuzz(func(t *testing.T, n, m, seed uint) {
		if n == 0 || m == 0 || n*m > 10000 {
			t.Skip()
		}

		x := make([]float32, n)
		w := make([]float32, n*m)

		rnd := rand.New(rand.NewSource(int64(seed)))
		fillRand(x, rnd)
		fillRand(w, rnd)

		o := make([]float32, m)
		nnfast.MatMul(o, x, w)

		o1 := make([]float32, m)
		nn.MatMul(o1, x, w)

		if !slices.Equal(o1, o) {
			t.Errorf("got %v, exp %v", o, o1)
		}
	})
}

func FuzzSampleTopP(f *testing.F) {
	tests := []struct {
		probabilities []float32
		topp          float32
		allowedIdexes map[int]bool
	}{
		{
			topp:          1.0,
			probabilities: []float32{0.2, 0.2, 0.2, 0.2, 0.2},
			allowedIdexes: map[int]bool{0: true, 1: true, 2: true, 3: true, 4: true},
		},
		{
			topp:          0.7,
			probabilities: []float32{0.05, 0.19, 0.21, 0.3, 0.25},
			allowedIdexes: map[int]bool{3: true, 4: true, 2: true},
		},
		{
			topp:          0.80,
			probabilities: []float32{0.5, 0.3, 0.19, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002},
			allowedIdexes: map[int]bool{0: true, 1: true, 2: true},
		},
		{
			topp:          0.81,
			probabilities: []float32{0.5, 0.3, 0.19, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002},
			allowedIdexes: map[int]bool{0: true, 1: true, 2: true},
		},
	}
	f.Fuzz(func(t *testing.T, n int) {
		for _, tc := range tests {
			var s float32
			for _, p := range tc.probabilities {
				s += p
			}
			if math.Abs(float64(s)-1) > 1e-5 {
				t.Fatalf("probabilities should sum up to 1, got %f (%#v)", s, s)
			}
			v := nnfast.SampleTopP(tc.probabilities, tc.topp)
			if !tc.allowedIdexes[v] {
				t.Errorf("unallowed index %d", v)
			}
		}
	})
}
