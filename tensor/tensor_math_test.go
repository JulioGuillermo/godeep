package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
)

func TestTSum(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Sum[float32](m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	sum := float32(0)
	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}
		}
	}

	f, err := r.Get(0)
	if f != sum {
		t.Fatal(f, "!=", sum)
	}
}

func TestTAvg(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Avg[float32](m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	sum := float32(0)
	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}
		}
	}
	sum /= float32(m.GetSize())

	f, err := r.Get(0)
	if f != sum {
		t.Fatal(f, "!=", sum)
	}
}

func TestTMax(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Max[float32](m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	sum := float32(0)
	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum < n {
					sum = n
				}
			}
		}
	}

	f, err := r.Get(0)
	if f != sum {
		t.Fatal(f, "!=", sum)
	}
}

func TestTMin(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Min[float32](m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	sum := float32(0)
	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum > n {
					sum = n
				}
			}
		}
	}

	f, err := r.Get(0)
	if f != sum {
		t.Fatal(f, "!=", sum)
	}
}
