package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
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
	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*operation.Operand[T], tools.GetDataSize(p.Shape))

	min := p.GetSize()
	if min > p.T.GetSize() {
		min = p.T.GetSize()
	}
	ops := p.T.GetOperands()
	for i := uint(0); i < min; i++ {
		p.Operands[i] = ops[i]
	}
	for i := min; i < p.GetSize(); i++ {
		p.Operands[i] = &operation.Operand[T]{}
	}

	return nil
}
