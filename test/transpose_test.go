package test

import (
	"testing"

	"github.com/julioguillermo/neuralnetwork/v2/graph"
	"github.com/julioguillermo/neuralnetwork/v2/tensor"
)

func TestTranspose(t *testing.T) {
	m := tensor.NewFromValues([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 3, 3)

	trans := tensor.Transpose[float32](m)

	g, err := graph.NewGraph(trans)
	if err != nil {
		t.Fatal(err)
	}
	g.Exec()
	if err != nil {
		t.Fatal(err)
	}

	t.Log(trans)

	for i := uint(0); i < m.GetShape()[0]; i++ {
		for j := uint(0); j < m.GetShape()[1]; j++ {
			v1, e := m.Get(i, j)
			if e != nil {
				t.Fatal(e)
			}
			v2, e := trans.Get(j, i)
			if e != nil {
				t.Fatal(e)
			}
			if v1 != v2 {
				t.Fatal(v1, v2)
			}
		}
	}
}
