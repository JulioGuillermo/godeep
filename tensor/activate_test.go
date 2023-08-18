package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestActivate(t *testing.T) {
	shape := []uint{50, 50, 20}

	act := &activation.Sin[float32]{}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Activate[float32](m, act.Activate)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing Activation", r.GetShape(), m.GetShape())
	if err != nil {
		t.Fatal(err)
	}

	mops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if o.Value != act.Activate(mops[i].Value) {
			t.Fatalf(
				"Operands at %d: %f != <act(%f) => %f>",
				i,
				o.Value,
				mops[i].Value,
				act.Activate(mops[i].Value),
			)
		}
	}
}

func TestHotActivate(t *testing.T) {
	shape := []uint{50, 50, 20}

	act := &activation.Sin[float32]{}

	m := tensor.NewNormRand[float32](shape...)
	r := m.Activate(act.Activate)

	mops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if o.Value != act.Activate(mops[i].Value) {
			t.Fatalf(
				"Operands at %d: %f != <act(%f) => %f>",
				i,
				o.Value,
				mops[i].Value,
				act.Activate(mops[i].Value),
			)
		}
	}
}
