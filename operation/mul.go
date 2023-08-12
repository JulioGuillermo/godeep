package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Mul[T types.Number] struct {
	*number.Scalar[T]
	A *number.Scalar[T]
	B *number.Scalar[T]
}

func (p *Mul[_]) Cal() {
	// p.Set(p.A.Get() * p.B.Get())
	p.Value = p.A.Value * p.B.Value
}
