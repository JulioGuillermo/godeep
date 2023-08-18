package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
)

func TestSet(t *testing.T) {
	r := number.NewScalar[float32](0)
	o := number.NewScalar[float32](-234)

	oper := &operation.Set[float32]{
		Scalar: r,
		O:      o,
	}

	oper.Cal()

	if r.Value != o.Value {
		t.Fatalf("Set fail: R(%f) != O(%f)", r.Value, o.Value)
	}
}
