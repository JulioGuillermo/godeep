package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestReshape(t *testing.T) {
	shape := []uint{50 * 50 * 20}
	nshape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Reshape[float32](m, nshape...)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing Reshape", r.GetShape(), nshape)
	if err != nil {
		t.Fatal(err)
	}

	mops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if mops[i].Value != o.Value {
			t.Fatalf("Operands at %d are differents: %f != %f", i, o.Value, mops[i].Value)
		}
	}
}

func TestHotReshape(t *testing.T) {
	shape := []uint{50 * 50 * 20}
	nshape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := m.Reshape(nshape...)

	err := tools.GetEqShapeErr("Testing Reshape", r.GetShape(), nshape)
	if err != nil {
		t.Fatal(err)
	}

	mops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if mops[i].Value != o.Value {
			t.Fatalf("Operands at %d are differents: %f != %f", i, o.Value, mops[i].Value)
		}
	}
}
