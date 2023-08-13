package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
)

func TestSum(t *testing.T) {
	r := tensor.NewScalar[float32](0)
	o1 := tensor.NewScalar[float32](-234)
	o2 := tensor.NewScalar[float32](762)
	o3 := tensor.NewScalar[float32](322)
	args := []*number.Scalar[float32]{o1, o2, o3}

	oper := &operation.Sum[float32]{
		Scalar: r,
		Args:   args,
	}

	oper.Cal()

	v := float32(0)
	for _, a := range args {
		v += a.Value
	}

	if r.Value != v {
		t.Fatalf("Sum fail: R(%f) != O(%f, %f, %f)<%f>", r.Value, o1.Value, o2.Value, o3.Value, v)
	}
}
