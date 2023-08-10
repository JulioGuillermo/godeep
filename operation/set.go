package operation

import "github.com/julioguillermo/neuralnetwork/v2/types"

type Set[T types.Number] struct {
	*Operand[T]
	O *Operand[T]
}

func (p *Set[_]) Cal() {
	p.Value = p.O.Value
	// p.Set(p.F(p.O.Get()))
}
