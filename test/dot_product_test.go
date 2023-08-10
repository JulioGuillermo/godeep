package test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
)

func TestDotProduct(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r := tensor.DotProduct(m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d1 := m1.GetData()
	d2 := m2.GetData()
	dp := float32(0)
	for i := range d1 {
		dp += d1[i] * d2[i]
	}

	res, err := r.Get(0)
	if err != nil {
		t.Fatal(err)
	}
	if res != dp {
		t.Fatal(res, "!=", dp)
	}
}
