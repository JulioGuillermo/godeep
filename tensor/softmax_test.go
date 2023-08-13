package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestSoftMax(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.SoftMax(m)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing Softmax", r.GetShape(), m.GetShape())
	if err != nil {
		t.Fatal(err)
	}

	mops := m.GetOperands()
	min := mops[0].Value
	for _, o := range mops {
		if min > o.Value {
			min = o.Value
		}
	}

	sum := float32(0)
	for _, o := range mops {
		sum += o.Value - min
	}

	for i, o := range r.GetOperands() {
		if (mops[i].Value-min)/sum != o.Value {
			t.Fatalf(
				"Operands at %d are differents: %f != ((%f-%f)/%f => %f)",
				i,
				o.Value,
				mops[i].Value,
				min,
				sum,
				mops[i].Value/sum,
			)
		}
	}
}
