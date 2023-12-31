package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
)

func TestSubTensor(t *testing.T) {
	shape := []uint{50, 50, 20}
	F := uint(20)
	T := uint(30)

	m1 := tensor.NewNormRand[float32](shape...)

	r := tensor.SubTensor[float32](m1, 1, F, T)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := F; j < T; j++ {
			for k := uint(0); k < shape[2]; k++ {
				r, _ := r.Get(i, j-F, k)
				m, _ := m1.Get(i, j, k)
				if r != m {
					t.Fatal(i, j, k, "=>", r, m)
				}
			}
		}
	}
}

func TestHotSubTensor(t *testing.T) {
	shape := []uint{50, 50, 20}
	F := uint(20)
	T := uint(30)

	m1 := tensor.NewNormRand[float32](shape...)

	r, err := m1.SubTensor(1, F, T)
	if err != nil {
		t.Fatal(err)
	}

	for i := uint(0); i < shape[0]; i++ {
		for j := F; j < T; j++ {
			for k := uint(0); k < shape[2]; k++ {
				r, _ := r.Get(i, j-F, k)
				m, _ := m1.Get(i, j, k)
				if r != m {
					t.Fatal(i, j, k, "=>", r, m)
				}
			}
		}
	}
}
