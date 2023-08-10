package test

import (
	"testing"

	"github.com/julioguillermo/neuralnetwork/v2/graph"
	"github.com/julioguillermo/neuralnetwork/v2/tensor"
)

func TestConcat(t *testing.T) {
	shape := []uint{50, 50, 20}

	m1 := tensor.NewNormRand[float32](shape...)
	m2 := tensor.NewNormRand[float32](shape...)

	r := tensor.Concat(m1, m2, 1)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]*2; j++ {
			for k := uint(0); k < shape[2]; k++ {
				if j < shape[1] {
					r, _ := r.Get(i, j, k)
					m, _ := m1.Get(i, j, k)
					if r != m {
						t.Fatal(i, j, k, "=>", r, m)
					}
				} else {
					r, _ := r.Get(i, j, k)
					m, _ := m2.Get(i, j-shape[1], k)
					if r != m {
						t.Fatal(i, j, k, "=>", r, m)
					}
				}
			}
		}
	}
}
