package activation

import (
	"math"

	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type Tanh[T types.Number] struct{}

func (Tanh[T]) Activate(t T) T {
	return T(math.Tanh(float64(t)))
}

func (Tanh[T]) Derive(t T) T {
	return T(1 / math.Pow(math.Cosh(float64(t)), 2))
}
