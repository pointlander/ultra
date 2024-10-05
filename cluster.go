// Copyright 2024 The Illuminatus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"runtime"

	"github.com/alixaxel/pagerank"
)

const (
	// Size is the link size
	Size = 4
	// Input is the network input size
	Input = Size
	// Scale is the scale of the search
	Scale = 33 //48 96
	// Width is the width of the result
	Width = 150
	// Samples is the number of samplee
	Samples = Scale * (Scale - 1) / 2
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []complex128
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...complex128) Matrix {
	if data == nil {
		data = make([]complex128, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// NewZeroMatrix creates a new float64 matrix of zeros
func NewZeroMatrix(cols, rows int) Matrix {
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, cols*rows),
	}
}

// RandomMatrix is a random matrix
type RandomMatrix struct {
	Cols int
	Rows int
	Seed int64
}

// NewRandomMatrix creates a new gaussian random matrix
func NewRandomMatrix(cols, rows int, seed int64) RandomMatrix {
	return RandomMatrix{
		Cols: cols,
		Rows: rows,
		Seed: seed,
	}
}

// Sample generates a matrix from a gaussian distribution
func (g RandomMatrix) Sample() Matrix {
	rng := rand.New(rand.NewSource(g.Seed))
	factor := math.Sqrt(2.0 / float64(g.Cols))
	sample := NewMatrix(g.Cols, g.Rows)
	for i := 0; i < g.Cols*g.Rows; i++ {
		a := rng.NormFloat64() * factor
		//b := rng.NormFloat64() * factor
		sample.Data = append(sample.Data, complex(a, 0))
	}
	return sample
}

// Dot computes the dot product
func Dot(x, y []complex128) (z complex128) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]complex128, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, Dot(mm, nn))
		}
	}
	return o
}

// PageRank computes the page rank of Q, K
func PageRank(x, y Matrix) []float64 {
	graph := pagerank.NewGraph()
	for i := 0; i < y.Rows; i++ {
		yy := y.Data[i*y.Cols : (i+1)*y.Cols]
		aa := complex(0.0, 0.0)
		for _, v := range yy {
			aa += v * v
		}
		aa = cmplx.Sqrt(aa)
		for j := 0; j < x.Rows; j++ {
			xx := x.Data[j*x.Cols : (j+1)*x.Cols]
			bb := complex(0.0, 0.0)
			for _, v := range xx {
				bb += v * v
			}
			bb = cmplx.Sqrt(bb)
			d := cmplx.Abs(Dot(yy, xx) / (aa * bb))
			graph.Link(uint32(i), uint32(j), d)
		}
	}
	ranks := make([]float64, y.Rows)
	graph.Rank(1.0, 1e-9, func(node uint32, rank float64) {
		ranks[node] = rank
	})
	return ranks
}

// Sample is a sample
type Sample struct {
	A     RandomMatrix
	B     RandomMatrix
	Ranks []float64
}

// Process processes the samples' the probability distribution
func Process(rng *rand.Rand, input Matrix, fisher []Fisher) {
	projections := make([]RandomMatrix, Scale)
	for i := range projections {
		seed := rng.Int63()
		if seed == 0 {
			seed = 1
		}
		projections[i] = NewRandomMatrix(Input, Input, seed)
	}
	index := 0
	samples := make([]Sample, Samples)
	for i := 0; i < Scale; i++ {
		for j := i + 1; j < Scale; j++ {
			samples[index].A = projections[i]
			samples[index].B = projections[j]
			index++
		}
	}

	done := make(chan bool, 8)
	process := func(sample *Sample) {
		a := sample.A.Sample()
		b := sample.B.Sample()
		x := a.MulT(input)
		y := b.MulT(input)
		sample.Ranks = PageRank(x, y)
		done <- true
	}
	flight, index, cpus := 0, 0, runtime.NumCPU()
	for flight < cpus && index < len(samples) {
		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for index < len(samples) {
		<-done
		flight--

		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for i := 0; i < flight; i++ {
		<-done
	}

	sums := make([]float64, Width)
	for i := range samples {
		for j := range samples[i].Ranks {
			sums[j] += samples[i].Ranks[j]
		}
	}
	averages := make([]float64, Width)
	for i := range sums {
		averages[i] = sums[i] / float64(len(samples))
	}
	variances := make([]float64, Width)
	for i := range averages {
		for j := range samples[i].Ranks {
			diff := averages[j] - samples[i].Ranks[j]
			variances[j] += diff * diff
		}
	}

	for i, item := range fisher {
		fmt.Println(item.Label, variances[i])
	}
}
