package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
)

func TestAbsN(t *testing.T) {
	r := number.NewScalar[float32](0)
	o := number.NewScalar[float32](-234)

	oper := &operation.Abs[float32]{
		Scalar: r,
		O:      o,
	}

	oper.Cal()

	v := o.Value
	if v < 0 {
		v = -v
	}
	if r.Value != v {
		t.Fatalf("Abs fail: R(%f) != O(%f)<%f>", r.Value, o.Value, v)
	}
}

func TestAbsP(t *testing.T) {
	r := number.NewScalar[float32](0)
	o := number.NewScalar[float32](234)

	oper := &operation.Abs[float32]{
		Scalar: r,
		O:      o,
	}

	oper.Cal()

	v := o.Value
	if v < 0 {
		v = -v
	}
	if r.Value != v {
		t.Fatalf("Abs fail: R(%f) != O(%f)<%f>", r.Value, o.Value, v)
	}
}
