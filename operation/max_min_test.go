package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
)

func TestMax(t *testing.T) {
	r := number.NewScalar[float32](0)
	o1 := number.NewScalar[float32](-234)
	o2 := number.NewScalar[float32](762)
	o3 := number.NewScalar[float32](322)
	args := []*number.Scalar[float32]{o1, o2, o3}

	oper := &operation.Max[float32]{
		Scalar: r,
		Args:   args,
	}

	oper.Cal()

	v := args[0].Value
	for _, a := range args {
		if v < a.Value {
			v = a.Value
		}
	}

	if r.Value != v {
		t.Fatalf("Max fail: R(%f) != O(%f, %f, %f)<%f>", r.Value, o1.Value, o2.Value, o3.Value, v)
	}
}

func TestMin(t *testing.T) {
	r := number.NewScalar[float32](0)
	o1 := number.NewScalar[float32](-234)
	o2 := number.NewScalar[float32](762)
	o3 := number.NewScalar[float32](322)
	args := []*number.Scalar[float32]{o1, o2, o3}

	oper := &operation.Min[float32]{
		Scalar: r,
		Args:   args,
	}

	oper.Cal()

	v := args[0].Value
	for _, a := range args {
		if v > a.Value {
			v = a.Value
		}
	}

	if r.Value != v {
		t.Fatalf("Min fail: R(%f) != O(%f, %f, %f)<%f>", r.Value, o1.Value, o2.Value, o3.Value, v)
	}
}
