package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Neg[T types.Number] struct {
	*number.Scalar[T]
	O *number.Scalar[T]
}

func (p *Neg[_]) Cal() {
	p.Value = -p.O.Value
}
