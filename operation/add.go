package operation

import "github.com/julioguillermo/godeep/types"

type Add[T types.Number] struct {
	*Operand[T]
	A *Operand[T]
	B *Operand[T]
}

func (p *Add[_]) Cal() {
	p.Value = p.A.Value + p.B.Value
	// p.Set(p.A.Get() + p.B.Get())
}
