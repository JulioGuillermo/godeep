package activation

import (
	"math"

	"github.com/julioguillermo/godeep/types"
)

type GELU[T types.Number] struct{}

func (GELU[T]) Activate(n T) T {
	x := float64(n)
	return T(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3)))))
}

func (p GELU[T]) Derive(x T) T {
	h := 0.000001
	H := T(h)
	if H == 0 {
		H++
	}
	return (p.Activate(x+H) - p.Activate(x-H)) / (2 * H)
}
