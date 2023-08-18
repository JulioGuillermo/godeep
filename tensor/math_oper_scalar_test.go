package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
)

func TestAddScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := number.NewScalar[float32](30)

	r := tensor.AddScalar[float32](m, s)
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
	s := number.NewScalar[float32](30)

	r := tensor.SubScalar[float32](m, s)
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
	s := number.NewScalar[float32](30)

	r := tensor.MulScalar[float32](m, s)
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
	s := number.NewScalar[float32](30)

	r := tensor.DivScalar[float32](m, s)
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

func TestHotAddScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := number.NewScalar[float32](30)

	r := m.AddScalar(s)

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]+s.Value {
			t.Fatal(v, "!=", d[i]+s.Value, "=>", d[i], s.Value)
		}
	}
}

func TestHotSubScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := number.NewScalar[float32](30)

	r := m.SubScalar(s)

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]-s.Value {
			t.Fatal(v, "!=", d[i]-s.Value, "=>", d[i], s.Value)
		}
	}
}

func TestHotMulScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := number.NewScalar[float32](30)

	r := m.MulScalar(s)

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]*s.Value {
			t.Fatal(v, "!=", d[i]*s.Value, "=>", d[i], s.Value)
		}
	}
}

func TestHotDivScalar(t *testing.T) {
	m := tensor.NewNormRand[float32](50, 50, 20)
	s := number.NewScalar[float32](30)

	r := m.DivScalar(s)

	d := m.GetData()
	for i, v := range r.GetData() {
		if v != d[i]/s.Value {
			t.Fatal(v, "!=", d[i]/s.Value, "=>", d[i], s.Value)
		}
	}
}
