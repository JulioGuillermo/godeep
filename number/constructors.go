package number

import (
	"math/rand"

	"github.com/julioguillermo/godeep/types"
)

func NewScalar[T types.Number](val T) *Scalar[T] {
	return &Scalar[T]{
		Value: val,
	}
}

func ScalarZero[T types.Number]() *Scalar[T] {
	return &Scalar[T]{}
}

func ScalarOne[T types.Number]() *Scalar[T] {
	return &Scalar[T]{
		Value: 1,
	}
}

func ScalarRand[T types.Number](min, max T) *Scalar[T] {
	return &Scalar[T]{
		Value: T(rand.Float64())*(max-min) + min,
	}
}

func ScalarNorm[T types.Number]() *Scalar[T] {
	return &Scalar[T]{
		Value: T(rand.NormFloat64()),
	}
}
