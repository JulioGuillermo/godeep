package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorReshape[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
}

func Reshape[T types.Number](t Tensor[T], shape ...uint) Tensor[T] {
	return &TensorReshape[T]{
		TensorMat: TensorMat[T]{
			Shape: shape,
		},
		T: t,
	}
}

func Flatten[T types.Number](t Tensor[T]) Tensor[T] {
	return Reshape(t, t.GetSize())
}

func (p *TensorReshape[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))

	min := p.GetSize()
	if min > p.T.GetSize() {
		min = p.T.GetSize()
	}
	ops := p.T.GetOperands()
	for i := uint(0); i < min; i++ {
		p.Operands[i] = ops[i]
	}
	for i := min; i < p.GetSize(); i++ {
		p.Operands[i] = &number.Scalar[T]{}
	}

	return nil
}

func (p *TensorMat[T]) Reshape(shape ...uint) *TensorMat[T] {
	mulIndex := tools.GetIndexMul(shape)
	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))

	min := p.GetSize()
	if min > uint(len(ops)) {
		min = uint(len(ops))
	}

	for i := uint(0); i < min; i++ {
		ops[i] = &number.Scalar[T]{
			Value: p.Operands[i].Value,
		}
	}

	for i := min; i < p.GetSize(); i++ {
		ops[i] = &number.Scalar[T]{}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}
}
