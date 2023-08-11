package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Layer[T types.Number] interface {
	GetInputs() tensor.Tensor[T]
	GetOutputs() tensor.Tensor[T]
	GetNetas() tensor.Tensor[T]
	GetDif() tensor.Tensor[T]
	GetActivation() activation.Activation[T]
	GetPrelayer() Layer[T]

	Connect(Layer[T])
	Build() error
	BuildFeedforward(*context.Context) error
	BuildBackpropagation(*context.Context, *operation.Operand[T], *operation.Operand[T]) error
	Fit() error

	SetTrainable(bool)
}
