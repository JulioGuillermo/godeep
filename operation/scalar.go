package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Scalar[T types.Number] struct {
	*number.Scalar[T]
}

// func (p *Scalar[_]) Lock()   {}
// func (p *Scalar[_]) Unlock() {}
func (p *Scalar[_]) Cal() {}
