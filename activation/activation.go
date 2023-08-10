package activation

import "github.com/julioguillermo/godeep/types"

type Activation[T types.Number] interface {
	Activate(T) T
	Derive(T) T
}
