package activation

import (
	"github.com/julioguillermo/godeep/types"
)

type SELU[T types.Number] struct {
	ELU[T]
	Lambda T
}

func (p SELU[T]) Activate(n T) T {
	return p.Lambda * p.ELU.Activate(n)
}

func (p SELU[T]) Derive(n T) T {
	return p.Lambda * p.ELU.Derive(n)
}
