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
	"strconv"

	"github.com/pointlander/ultra/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

// Cluster clusters the data
func Cluster(k int) {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}
	_ = data

	type Fisher struct {
		Measures [4]float64
		Label    string
		Cluster  int
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
			for _, item := range data {
				record := Fisher{
					Label: item[4],
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
	input := make([][]float64, 0, 8)
	for _, item := range fisher {
		input = append(input, item.Measures[:])
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
}

func main() {
	for i := 1; i < 8; i++ {
		fmt.Println("Cluster", i)
		Cluster(i)
	}
}
