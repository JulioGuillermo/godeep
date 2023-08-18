package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
)

func TestAdd(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r := tensor.Add[float32](m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]+d2[i] {
			t.Fatal(s, "!=", d1[i]+d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestSub(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r := tensor.Sub[float32](m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]-d2[i] {
			t.Fatal(s, "!=", d1[i]-d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestMul(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r := tensor.Mul[float32](m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]*d2[i] {
			t.Fatal(s, "!=", d1[i]*d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestDiv(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r := tensor.Div[float32](m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]/d2[i] {
			t.Fatal(s, "!=", d1[i]/d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestHotAdd(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r, err := m1.Add(m2)
	if err != nil {
		t.Fatal(err)
	}

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]+d2[i] {
			t.Fatal(s, "!=", d1[i]+d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestHotSub(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r, err := m1.Sub(m2)
	if err != nil {
		t.Fatal(err)
	}

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]-d2[i] {
			t.Fatal(s, "!=", d1[i]-d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestHotMul(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r, err := m1.Mul(m2)
	if err != nil {
		t.Fatal(err)
	}

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]*d2[i] {
			t.Fatal(s, "!=", d1[i]*d2[i], "=>", d1[i], d2[i])
		}
	}
}

func TestHotDiv(t *testing.T) {
	m1 := tensor.NewNormRand[float32](50, 50, 20)
	m2 := tensor.NewNormRand[float32](50, 50, 20)

	r, err := m1.Div(m2)
	if err != nil {
		t.Fatal(err)
	}

	d1 := m1.GetData()
	d2 := m2.GetData()
	for i, s := range r.GetData() {
		if s != d1[i]/d2[i] {
			t.Fatal(s, "!=", d1[i]/d2[i], "=>", d1[i], d2[i])
		}
	}
}
