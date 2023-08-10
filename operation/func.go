package operation

import "github.com/julioguillermo/godeep/types"

type Function[T types.Number] func(T) T

type Func[T types.Number] struct {
	*Operand[T]
	O *Operand[T]
	F Function[T]
}

func (p *Func[_]) Cal() {
	p.Value = p.F(p.O.Value)
	// p.Set(p.F(p.O.Get()))
}
