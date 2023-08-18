package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestDotProductVecVec(t *testing.T) {
	t.Log("Vec Vec")

	v1 := tensor.NewFromValues[float32]([]float32{1, 2, 9, 8}, 4)
	v2 := tensor.NewFromValues[float32]([]float32{4, 3, 5, 8}, 4)
	var R float32 = 119

	r := tensor.Dot[float32](v1, v2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	n, err := r.Get(0)
	if err != nil {
		t.Fatal(err)
	}
	if n != R {
		t.Errorf("Not good => %f and %f", n, R)
	}
}

func TestDotProductMatVec(t *testing.T) {
	t.Log("Mat Vec")

	v := tensor.NewFromValues[float32]([]float32{1, 2}, 2)
	m := tensor.NewFromValues[float32]([]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4}, 2, 3, 2)
	R := []float32{16, 25, 11, 17, 11, 9}

	r := tensor.Dot[float32](m, v)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	s1 := m.GetShape()
	s2 := r.GetShape()
	for i, s := range s2 {
		if s != s1[i] {
			t.Errorf("Not good: shape at %d => %d and %d", i, s, s1[i])
		}
	}

	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

func TestDotProductVecMat(t *testing.T) {
	t.Log("Vec Mat")

	v := tensor.NewFromValues[float32]([]float32{1, 2, 3}, 3)
	m := tensor.NewFromValues[float32]([]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4}, 2, 3, 2)
	R := []float32{33, 33, 22, 22}

	r := tensor.Dot[float32](v, m)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	s1 := m.GetShape()
	s1[len(s1)-2] = s1[len(s1)-1]
	s2 := r.GetShape()
	for i, s := range s2 {
		if s != s1[i] {
			t.Errorf("Not good: shape at %d => %d and %d", i, s, s1[i])
		}
	}

	t.Log(R)
	t.Log(r)
	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

func TestDotProductMatMat(t *testing.T) {
	t.Log("Mat Mat")

	m1 := tensor.NewFromValues[float32]([]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4}, 2, 3, 2)
	m2 := tensor.NewFromValues[float32](
		[]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4, 2, 2, 5, 4},
		4,
		2,
		2,
	)
	R := []float32{
		81, 70, 43, 54, 47, 32, 37, 32, 126, 109,
		67, 84, 71, 50, 58, 50, 54, 47, 29, 36,
		25, 22, 26, 22, 84, 73, 45, 56, 41, 34,
		40, 34, 60, 51, 31, 40, 51, 22, 24, 22,
		42, 37, 23, 28, 11, 18, 22, 18,
	}

	r := tensor.Dot[float32](m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing dot product result shape", r.GetShape(), []uint{2, 3, 4, 2})
	if err != nil {
		t.Fatal(err)
	}

	t.Log(R)
	t.Log(r)
	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

func TestDotProductMatMat2(t *testing.T) {
	t.Log("Mat Mat")

	m1 := tensor.NewFromValues[float32]([]float32{6, 9, 8, 9, 8, 7}, 3, 2)
	m2 := tensor.NewFromValues[float32]([]float32{3, 4, 5, 6}, 2, 2)
	R := []float32{63, 78, 69, 86, 59, 74}

	r := tensor.Dot[float32](m1, m2)
	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	err = tools.GetEqShapeErr("Testing dot product result shape", r.GetShape(), []uint{3, 2})
	if err != nil {
		t.Fatal(err)
	}

	t.Log(R)
	t.Log(r)
	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

// Hot

func TestHotDotProductVecVec(t *testing.T) {
	t.Log("Vec Vec")

	v1 := tensor.NewFromValues[float32]([]float32{1, 2, 9, 8}, 4)
	v2 := tensor.NewFromValues[float32]([]float32{4, 3, 5, 8}, 4)
	var R float32 = 119

	r, err := v1.Dot(v2)
	if err != nil {
		t.Fatal(err)
	}

	n, err := r.Get(0)
	if err != nil {
		t.Fatal(err)
	}
	if n != R {
		t.Errorf("Not good => %f and %f", n, R)
	}
}

func TestHotDotProductMatVec(t *testing.T) {
	t.Log("Mat Vec")

	v := tensor.NewFromValues[float32]([]float32{1, 2}, 2)
	m := tensor.NewFromValues[float32]([]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4}, 2, 3, 2)
	R := []float32{16, 25, 11, 17, 11, 9}

	r, err := m.Dot(v)
	if err != nil {
		t.Fatal(err)
	}

	s1 := m.GetShape()
	s2 := r.GetShape()
	for i, s := range s2 {
		if s != s1[i] {
			t.Errorf("Not good: shape at %d => %d and %d", i, s, s1[i])
		}
	}

	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

func TestHotDotProductVecMat(t *testing.T) {
	t.Log("Vec Mat")

	v := tensor.NewFromValues[float32]([]float32{1, 2, 3}, 3)
	m := tensor.NewFromValues[float32]([]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4}, 2, 3, 2)
	R := []float32{33, 33, 22, 22}

	r, err := v.Dot(m)
	if err != nil {
		t.Fatal(err)
	}

	s1 := m.GetShape()
	s1[len(s1)-2] = s1[len(s1)-1]
	s2 := r.GetShape()
	for i, s := range s2 {
		if s != s1[i] {
			t.Errorf("Not good: shape at %d => %d and %d", i, s, s1[i])
		}
	}

	t.Log(R)
	t.Log(r)
	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

func TestHotDotProductMatMat(t *testing.T) {
	t.Log("Mat Mat")

	m1 := tensor.NewFromValues[float32]([]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4}, 2, 3, 2)
	m2 := tensor.NewFromValues[float32](
		[]float32{6, 5, 9, 8, 3, 4, 5, 6, 7, 2, 1, 4, 2, 2, 5, 4},
		4,
		2,
		2,
	)
	R := []float32{
		81, 70, 43, 54, 47, 32, 37, 32, 126, 109,
		67, 84, 71, 50, 58, 50, 54, 47, 29, 36,
		25, 22, 26, 22, 84, 73, 45, 56, 41, 34,
		40, 34, 60, 51, 31, 40, 51, 22, 24, 22,
		42, 37, 23, 28, 11, 18, 22, 18,
	}

	r, err := m1.Dot(m2)
	if err != nil {
		t.Fatal(err)
	}

	err = tools.GetEqShapeErr("Testing dot product result shape", r.GetShape(), []uint{2, 3, 4, 2})
	if err != nil {
		t.Fatal(err)
	}

	t.Log(R)
	t.Log(r)
	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}

func TestHotDotProductMatMat2(t *testing.T) {
	t.Log("Mat Mat")

	m1 := tensor.NewFromValues[float32]([]float32{6, 9, 8, 9, 8, 7}, 3, 2)
	m2 := tensor.NewFromValues[float32]([]float32{3, 4, 5, 6}, 2, 2)
	R := []float32{63, 78, 69, 86, 59, 74}

	r, err := m1.Dot(m2)
	if err != nil {
		t.Fatal(err)
	}

	err = tools.GetEqShapeErr("Testing dot product result shape", r.GetShape(), []uint{3, 2})
	if err != nil {
		t.Fatal(err)
	}

	t.Log(R)
	t.Log(r)
	for i, o := range r.GetOperands() {
		if o.Value != R[i] {
			t.Errorf("Not good at %d => %f and %f", i, o.Value, R[i])
		}
	}
}
