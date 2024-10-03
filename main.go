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
	"strconv"

	"github.com/pointlander/ultra/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

func main() {
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
		clusters, _, err := kmeans.Kmeans(int64(i+1), input, 3, kmeans.SquaredEuclideanDistance, -1)
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
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for key, value := range clusters {
		fisher[key].Cluster = value
	}
	for _, v := range fisher {
		fmt.Println(v.Label, v.Cluster)
	}
}
