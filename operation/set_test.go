package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
)

func TestSet(t *testing.T) {
	r := tensor.NewScalar[float32](0)
	o := tensor.NewScalar[float32](-234)

	oper := &operation.Set[float32]{
		Scalar: r,
		O:      o,
	}

	oper.Cal()

	if r.Value != o.Value {
		t.Fatalf("Set fail: R(%f) != O(%f)", r.Value, o.Value)
	}
}
