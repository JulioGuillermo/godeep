package activation

import (
	"math"

	"github.com/julioguillermo/godeep/types"
)

type Sigmoid[T types.Number] struct{}

func (Sigmoid[T]) Activate(t T) T {
	return 1 / (1 + T(math.Exp(float64(-t))))
}

func (p Sigmoid[T]) Derive(t T) T {
	a := p.Activate(t)
	return a * (1.0 - a)
}
