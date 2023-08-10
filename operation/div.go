package operation

import "github.com/julioguillermo/neuralnetwork/v2/types"

type Div[T types.Number] struct {
	*Operand[T]
	A *Operand[T]
	B *Operand[T]
}

func (p *Div[_]) Cal() {
	// p.Set(p.A.Get() / p.B.Get())
	p.Value = p.A.Value / p.B.Value
}
