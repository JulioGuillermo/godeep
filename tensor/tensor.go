package tensor

import (
	"io"

	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Tensor[T types.Number] interface {
	GetShape() []uint
	GetMulIndex() []uint
	GetSize() uint
	GetOperands() []*number.Scalar[T]
	GetOperand(...uint) (*number.Scalar[T], error)
	GetMatrix() *TensorMat[T]

	GetData() []T
	LoadData([]T) error
	LoadFromTensor(Tensor[T]) error
	Bind(Tensor[T]) error

	Get(...uint) (T, error)
	Set(T, ...uint) error
	String() string
	Copy() Tensor[T]

	BuildGraph(*context.Context) error

	SetBuild(bool) Tensor[T]

	Save(io.Writer) error
	SaveFull(io.Writer) error
	Load(io.Reader) error
	LoadFull(io.Reader) error
}
