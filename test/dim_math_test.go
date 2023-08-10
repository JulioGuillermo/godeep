package test

import (
	"testing"

	"github.com/julioguillermo/neuralnetwork/v2/graph"
	"github.com/julioguillermo/neuralnetwork/v2/tensor"
)

func TestDimSum(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DSum(m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestDimAvg(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DAvg(m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}
			sum /= float32(shape[2])

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestDimMax(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DMax(m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			n, err := m.Get(i, j, 0)
			if err != nil {
				t.Fatal(err)
			}
			sum := n
			for k := uint(1); k < shape[2]; k++ {
				n, err = m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum < n {
					sum = n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestDimMin(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DMin(m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			n, err := m.Get(i, j, 0)
			if err != nil {
				t.Fatal(err)
			}
			sum := n
			for k := uint(1); k < shape[2]; k++ {
				n, err = m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum > n {
					sum = n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}
