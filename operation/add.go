package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Add[T types.Number] struct {
	*number.Scalar[T]
	A *number.Scalar[T]
	B *number.Scalar[T]
}

func (p *Add[_]) Cal() {
	p.Value = p.A.Value + p.B.Value
	// p.Set(p.A.Get() + p.B.Get())
}
