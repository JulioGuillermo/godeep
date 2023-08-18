package tools

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

func Sum[T types.Number](args []*number.Scalar[T]) T {
	sum := T(0)
	for _, a := range args {
		sum += a.Value
	}
	return sum
}

func Avg[T types.Number](args []*number.Scalar[T]) T {
	sum := T(0)
	for _, a := range args {
		sum += a.Value
	}
	return sum / T(len(args))
}

func Max[T types.Number](args []*number.Scalar[T]) T {
	max := args[0].Value
	for i := 1; i < len(args); i++ {
		if max < args[i].Value {
			max = args[i].Value
		}
	}
	return max
}

func Min[T types.Number](args []*number.Scalar[T]) T {
	min := args[0].Value
	for i := 1; i < len(args); i++ {
		if min > args[i].Value {
			min = args[i].Value
		}
	}
	return min
}
