package operation

import "github.com/julioguillermo/godeep/types"

type Scalar[T types.Number] struct {
	*Operand[T]
}

// func (p *Scalar[_]) Lock()   {}
// func (p *Scalar[_]) Unlock() {}
func (p *Scalar[_]) Cal() {}
