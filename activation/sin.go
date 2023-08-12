package activation

import (
	"math"

	"github.com/julioguillermo/godeep/types"
)

type Sin[T types.Number] struct{}

func (Sin[T]) Activate(t T) T {
	return T(math.Sin(float64(t)))
}

func (Sin[T]) Derive(t T) T {
	return T(math.Cos(float64(t)))
}
