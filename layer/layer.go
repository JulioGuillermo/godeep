package layer

import (
	"strings"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Layer[T types.Number] interface {
	GetIndex() uint
	GetInputs() tensor.Tensor[T]
	GetOutputs() tensor.Tensor[T]
	GetNetas() tensor.Tensor[T]
	GetDif() tensor.Tensor[T]
	GetRef() *number.Scalar[T]
	GetActivation() activation.Activation[T]
	GetPrelayer() Layer[T]

	Connect(Layer[T])
	Build() (uint, error)
	BuildFeedforward(*context.Context) error
	BuildBackpropagation(*context.Context, *number.Scalar[T], *number.Scalar[T]) error
	Fit() error
	ResetFit(*context.Context) error
	Reset(*context.Context) error

	ResetPrinted()

	PushToString(sb *strings.Builder)
	String() string

	SetTrainable(bool)
}
