package operation

import "github.com/julioguillermo/godeep/types"

type Neg[T types.Number] struct {
	*Operand[T]
	O *Operand[T]
}

func (p *Neg[_]) Cal() {
	p.Value = -p.O.Value
}
