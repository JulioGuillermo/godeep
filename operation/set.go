package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Set[T types.Number] struct {
	*number.Scalar[T]
	O *number.Scalar[T]
}

func (p *Set[_]) Cal() {
	p.Value = p.O.Value
	// p.Set(p.F(p.O.Get()))
}
