package activation

import (
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type Activation[T types.Number] interface {
	Activate(T) T
	Derive(T) T
}
