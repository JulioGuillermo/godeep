package activation

import (
	"math"

	"github.com/julioguillermo/godeep/types"
)

type ELU[T types.Number] struct {
	Alpha T
}

func (p ELU[T]) Activate(n T) T {
	if n >= 0 {
		return n
	}
	return p.Alpha * (T(math.Exp(float64(n)) - 1))
}

func (p ELU[T]) Derive(n T) T {
	if n >= 0 {
		return n
	}
	return p.Activate(n) + p.Alpha
}
