package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
)

func TestSub(t *testing.T) {
	r := tensor.NewScalar[float32](0)
	a := tensor.NewScalar[float32](234)
	b := tensor.NewScalar[float32](876)

	oper := &operation.Sub[float32]{
		Scalar: r,
		A:      a,
		B:      b,
	}

	oper.Cal()

	v := a.Value - b.Value
	if r.Value != v {
		t.Fatalf("Sub fail: R(%f) != O(%f - %f)<%f>", r.Value, a.Value, b.Value, v)
	}
}
