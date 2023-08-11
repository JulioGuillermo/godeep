package activation

import "github.com/julioguillermo/godeep/types"

type Linear[T types.Number] struct{}

func (Linear[T]) Activate(n T) T {
	return n
}

func (Linear[T]) Derive(T) T {
	return 1
}
