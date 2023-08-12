package activation

import (
	"github.com/julioguillermo/godeep/types"
)

type Relu[T types.Number] struct {
	M T
}

func (p Relu[T]) Activate(t T) T {
	if t < 0 {
		return t * p.M
	}
	return t
}

func (p Relu[T]) Derive(t T) T {
	if t < 0 {
		return p.M
	}
	return 1
}
