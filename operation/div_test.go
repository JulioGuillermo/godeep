package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
)

func TestDiv(t *testing.T) {
	r := number.NewScalar[float32](0)
	a := number.NewScalar[float32](234)
	b := number.NewScalar[float32](876)

	oper := &operation.Div[float32]{
		Scalar: r,
		A:      a,
		B:      b,
	}

	oper.Cal()

	v := a.Value / b.Value
	if r.Value != v {
		t.Fatalf("Div fail: R(%f) != O(%f / %f)<%f>", r.Value, a.Value, b.Value, v)
	}
}
