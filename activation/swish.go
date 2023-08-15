package activation

import (
	"github.com/julioguillermo/godeep/types"
)

type Swish[T types.Number] struct {
	Sigmoid[T]
}

func (p Swish[T]) Activate(t T) T {
	return t * p.Sigmoid.Activate(t)
}

func (p Swish[T]) Derive(t T) T {
	return p.Activate(t) + p.Sigmoid.Activate(t)*(1-p.Activate(t))
}
