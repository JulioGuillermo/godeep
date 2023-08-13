package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
)

func TestNeg(t *testing.T) {
	r := tensor.NewScalar[float32](0)
	o := tensor.NewScalar[float32](-234)

	oper := &operation.Neg[float32]{
		Scalar: r,
		O:      o,
	}

	oper.Cal()

	v := -o.Value
	if r.Value != v {
		t.Fatalf("Neg fail: R(%f) != O(%f)<%f>", r.Value, o.Value, v)
	}
}
