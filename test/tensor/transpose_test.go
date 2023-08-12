package test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestTranspose(t *testing.T) {
	m := tensor.NewFromValues(
		[]float32{2, 3, 4, 6, 7, 5, 3, 2, 5, 85, 4, 7, 7, 8, 9, 5, 8, 5, 6, 4, 3, 5, 7, 4},
		1,
		3,
		2,
		4,
	)
	R := []float32{2, 5, 8, 7, 7, 3, 3, 85, 5, 5, 8, 5, 4, 4, 6, 3, 9, 7, 6, 7, 4, 2, 5, 4}

	trans := tensor.Transpose[float32](m)

	g, err := graph.NewGraph(trans)
	if err != nil {
		t.Fatal(err)
	}
	g.Exec()

	t.Log(trans)

	err = tools.GetEqShapeErr(
		"Testing transpose result shape",
		trans.GetShape(),
		[]uint{4, 2, 3, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	for i, o := range trans.GetData() {
		if o != R[i] {
			t.Fatalf("Fail at %d => %f and %f", i, R[i], o)
		}
	}
}
