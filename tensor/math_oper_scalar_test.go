package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
)

func TestAddScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := tensor.NewScalar[float32](30)

	r := tensor.AddScalar(m, s)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]+s.Value {
			t.Fatal(v, "!=", d[i]+s.Value, "=>", d[i], s.Value)
		}
	}
}

func TestSubScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := tensor.NewScalar[float32](30)

	r := tensor.SubScalar(m, s)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]-s.Value {
			t.Fatal(v, "!=", d[i]-s.Value, "=>", d[i], s.Value)
		}
	}
}

func TestMulScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := tensor.NewScalar[float32](30)

	r := tensor.MulScalar(m, s)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]*s.Value {
			t.Fatal(v, "!=", d[i]*s.Value, "=>", d[i], s.Value)
		}
	}
}

func TestDivScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := tensor.NewScalar[float32](30)

	r := tensor.DivScalar(m, s)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]/s.Value {
			t.Fatal(v, "!=", d[i]/s.Value, "=>", d[i], s.Value)
		}
	}
}
