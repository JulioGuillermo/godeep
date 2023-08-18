package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestNeg(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Neg[float32](m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing Neg", r.GetShape(), m.GetShape())
	if err != nil {
		t.Fatal(err)
	}

	ops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if o.Value != -ops[i].Value {
			t.Fatalf(
				"Operands at %d are differents: %f != (%f => %f)",
				i,
				o.Value,
				ops[i].Value,
				-ops[i].Value,
			)
		}
	}
}

func TestHotNeg(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := m.Neg()

	err := tools.GetEqShapeErr("Testing Neg", r.GetShape(), m.GetShape())
	if err != nil {
		t.Fatal(err)
	}

	ops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if o.Value != -ops[i].Value {
			t.Fatalf(
				"Operands at %d are differents: %f != (%f => %f)",
				i,
				o.Value,
				ops[i].Value,
				-ops[i].Value,
			)
		}
	}
}
