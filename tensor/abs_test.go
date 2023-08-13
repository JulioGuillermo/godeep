package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestAbs(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.Abs(m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing Abs", r.GetShape(), m.GetShape())
	if err != nil {
		t.Fatal(err)
	}

	mops := m.GetOperands()
	for i, o := range r.GetOperands() {
		if (mops[i].Value < 0 && -mops[i].Value != o.Value) ||
			(mops[i].Value >= 0 && mops[i].Value != o.Value) {
			t.Fatalf("Operands at %d are differents: %f != %f", i, o.Value, mops[i].Value)
		}
	}
}
