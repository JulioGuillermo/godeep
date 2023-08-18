package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
)

func TestAvg(t *testing.T) {
	r := number.NewScalar[float32](0)
	o1 := number.NewScalar[float32](-234)
	o2 := number.NewScalar[float32](762)
	o3 := number.NewScalar[float32](322)
	args := []*number.Scalar[float32]{o1, o2, o3}

	oper := &operation.Avg[float32]{
		Scalar: r,
		Args:   args,
	}

	oper.Cal()

	v := float32(0)
	for _, a := range args {
		v += a.Value
	}
	v /= float32(len(args))

	if r.Value != v {
		t.Fatalf("Avg fail: R(%f) != O(%f, %f, %f)<%f>", r.Value, o1.Value, o2.Value, o3.Value, v)
	}
}
