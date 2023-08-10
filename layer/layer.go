package layer

import (
	"github.com/julioguillermo/neuralnetwork/v2/activation"
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tensor"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type Layer[T types.Number] interface {
	GetInputs() tensor.Tensor[T]
	GetOutputs() tensor.Tensor[T]
	GetNetas() tensor.Tensor[T]
	GetDif() tensor.Tensor[T]
	GetActivation() activation.Activation[T]

	Conect(Layer[T])
	Build() error
	BuildFeedforward(*context.Context) error
	BuildBackpropagation(*context.Context, *operation.Operand[T], *operation.Operand[T]) error
	Fit() error

	SetTrainable(bool)
}
