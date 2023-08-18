package activation

import (
	"github.com/julioguillermo/godeep/types"
)

type Relu[T types.Number] struct {
	Alpha T
}

func (p Relu[T]) Activate(t T) T {
	if t < 0 {
		return t * p.Alpha
	}
	return t
}

func (p Relu[T]) Derive(t T) T {
	if t < 0 {
		return p.Alpha
	}
	return 1
}
