package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Equals[T types.Number] struct {
	*number.Scalar[T]
	A *number.Scalar[T]
	B *number.Scalar[T]
}

func (p *Equals[_]) Cal() {
	if p.A.Value == p.B.Value {
		p.Value = 1
	} else {
		p.Value = 0
	}
}
