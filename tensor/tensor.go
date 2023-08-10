package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type Tensor[T types.Number] interface {
	GetShape() []uint
	GetMulIndex() []uint
	GetSize() uint
	GetOperands() []*operation.Operand[T]
	GetOperand(...uint) (*operation.Operand[T], error)

	GetData() []T
	LoadData([]T) error
	LoadFromTensor(Tensor[T]) error
	Bind(Tensor[T]) error

	Get(...uint) (T, error)
	Set(T, ...uint) error
	String() string
	Copy() Tensor[T]

	BuildGraph(*context.Context) error
}
