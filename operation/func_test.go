package operation_test

import (
	"testing"

	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
)

func f(x float32) float32 {
	return 1.0 / x
}

func TestFunc(t *testing.T) {
	r := tensor.NewScalar[float32](0)
	o := tensor.NewScalar[float32](-234)

	oper := &operation.Func[float32]{
		Scalar: r,
		O:      o,
		F:      f,
	}

	oper.Cal()

	v := f(o.Value)
	if r.Value != v {
		t.Fatalf("Neg fail: R(%f) != O(%f)<%f>", r.Value, o.Value, v)
	}
}
