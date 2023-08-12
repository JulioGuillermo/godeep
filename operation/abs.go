package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Abs[T types.Number] struct {
	*number.Scalar[T]
	O *number.Scalar[T]
}

func (p *Abs[_]) Cal() {
	if p.O.Value < 0 {
		p.Value = -p.O.Value
	} else {
		p.Value = p.O.Value
	}
}
