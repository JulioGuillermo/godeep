package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestFill(t *testing.T) {
	shape := []uint{50, 50, 20}

	s := tensor.NewScalar[float32](3.14)
	m := tensor.NewNormRand[float32](shape...)
	r := tensor.FillWith(m, s)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing Fill", r.GetShape(), m.GetShape())
	if err != nil {
		t.Fatal(err)
	}

	for i, o := range r.GetOperands() {
		if o.Value != s.Value {
			t.Fatalf("Operands at %d are differents: %f != %f", i, o.Value, s.Value)
		}
	}
}
