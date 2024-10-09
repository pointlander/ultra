// Copyright 2024 The Ultra Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"

	"github.com/pointlander/ultra/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for i := range item[:4] {
					f, err := strconv.ParseFloat(item[i], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[i] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// Entropy calculates the entropy of the clustering
func Entropy(fisher []Fisher, c int, clusters []int) {
	ab := make([][]float64, 3)
	for i := range ab {
		ab[i] = make([]float64, c)
	}
	ba := make([][]float64, c)
	for i := range ba {
		ba[i] = make([]float64, 3)
	}
	for i := range fisher {
		a := int(Labels[fisher[i].Label])
		b := clusters[i]
		ab[a][b]++
		ba[b][a]++
	}
	entropy := 0.0
	for i := 0; i < c; i++ {
		entropy += (1.0 / float64(c)) * math.Log(1.0/float64(c))
	}
	fmt.Println(-entropy, -(1.0/float64(c))*math.Log(1.0/float64(c)))
	for i := range ab {
		entropy := 0.0
		for _, value := range ab[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		fmt.Println("ab", i, entropy)
	}
	for i := range ba {
		entropy := 0.0
		for _, value := range ba[i] {
			if value > 0 {
				p := value / 150
				entropy += p * math.Log(p)
			}
		}
		entropy = -entropy
		fmt.Println("ba", i, entropy)
	}
}

// Cluster clusters the data
func Cluster(k int, vars [][]float64) []int {
	fisher := Load()
	input := make([][]float64, 0, 8)
	for i, item := range fisher {
		measures := make([]float64, len(vars))
		for j := range measures {
			measures[j] = vars[j][i]
		}
		item.Measures = measures
		input = append(input, item.Measures)
	}
	meta := make([][]float64, len(fisher))
	for i := range meta {
		meta[i] = make([]float64, len(fisher))
	}
	for i := 0; i < 100; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), input, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, k, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for key, value := range clusters {
		fisher[key].Cluster = value
	}
	sum, counts := make([][4]float64, k), make([]float64, k)
	for _, v := range fisher {
		for i, vv := range v.Measures {
			sum[v.Cluster][i] += vv
		}
		counts[v.Cluster]++
		//fmt.Println(v.Label, v.Cluster)
	}
	for i := range sum {
		for j := range sum[i] {
			sum[i][j] /= counts[i]
		}
	}
	stddev := make([][4]float64, k)
	for _, v := range fisher {
		for i, vv := range v.Measures {
			diff := sum[v.Cluster][i] - vv
			stddev[v.Cluster][i] += diff * diff
		}
	}
	for i := range stddev {
		for j := range stddev[i] {
			stddev[i][j] = math.Sqrt(stddev[i][j] / counts[i])
		}
	}
	outliers := make([][4]int, k)
	for _, v := range fisher {
		for j, vv := range v.Measures {
			if math.Abs(vv-sum[v.Cluster][j]) > 3*stddev[v.Cluster][j] {
				outliers[v.Cluster][j]++
			}
		}
	}
	fmt.Println(outliers)
	total := 0
	for _, v := range outliers {
		for _, vv := range v {
			total += vv
		}
	}
	fmt.Println("total", total)
	return clusters
}

func Split(fisher []Fisher, col int) (float64, int) {
	sort.Slice(fisher, func(i, j int) bool {
		return fisher[i].Measures[col] < fisher[j].Measures[col]
	})
	sum := 0.0
	for _, item := range fisher {
		sum += item.Measures[col]
	}
	average := sum / float64(len(fisher))
	variance := 0.0
	for _, item := range fisher {
		diff := item.Measures[col] - average
		variance += diff * diff
	}
	variance /= float64(len(fisher))
	max, index := 0.0, 0
	for i := 1; i < len(fisher)-1; i++ {
		sumA, sumB := 0.0, 0.0
		countA, countB := 0.0, 0.0
		for _, item := range fisher[:i] {
			sumA += item.Measures[col]
			countA++
		}
		for _, item := range fisher[i:] {
			sumB += item.Measures[col]
			countB++
		}
		averageA := sumA / countA
		averageB := sumB / countB
		varianceA, varianceB := 0.0, 0.0
		for _, item := range fisher[:i] {
			diff := item.Measures[col] - averageA
			varianceA += diff * diff
		}
		for _, item := range fisher[i:] {
			diff := item.Measures[col] - averageB
			varianceB += diff * diff
		}
		varianceA /= countA
		varianceB /= countB
		if diff := variance - (varianceA + varianceB); diff > max {
			max, index = diff, i
		}
	}
	return max, index
}

func main() {
	rng := rand.New(rand.NewSource(1))
	fisher := Load()
	vars := make([][]float64, 0, 8)
	for i := 0; i < 4; i++ {
		input := NewMatrix(4+i, 150)
		for i := range fisher {
			for _, value := range fisher[i].Measures {
				input.Data = append(input.Data, complex(value, 0))
			}
		}
		variances := Process(rng, input, fisher)
		for i := range fisher {
			fisher[i].Measures = append(fisher[i].Measures, variances[i])
		}
		vars = append(vars, variances)
	}

	for i := 1; i < 8; i++ {
		fmt.Println("Cluster", i)
		clusters := Cluster(i, vars)
		Entropy(fisher, i, clusters)
	}

	cluster := func(fisher []Fisher, col int) []int {
		clusters := make([]int, len(fisher))
		_, index := Split(fisher, col)
		max1, index1 := Split(fisher[:index], col)
		max2, index2 := Split(fisher[index:], col)
		if max1 > max2 {
			for i, item := range fisher {
				if i < index1 {
					clusters[item.Index] = 0
				} else if i < index {
					clusters[item.Index] = 1
				} else {
					clusters[item.Index] = 2
				}
			}
		} else {
			for i, item := range fisher {
				if i < index {
					clusters[item.Index] = 0
				} else if i < index+index2 {
					clusters[item.Index] = 1
				} else {
					clusters[item.Index] = 2
				}
			}
		}
		return clusters
	}
	meta := make([][]float64, len(fisher))
	for i := range meta {
		meta[i] = make([]float64, len(fisher))
	}
	for i := 4; i < 8; i++ {
		clusters := cluster(fisher, i)
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	fisher = Load()
	for i, v := range clusters {
		fmt.Println(fisher[i].Label, v)
	}
	Entropy(fisher, 3, clusters)
}
