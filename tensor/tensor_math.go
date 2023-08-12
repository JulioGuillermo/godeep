package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type OperTensor byte

const (
	TensorSum = OperTensor(iota)
	TensorAvg
	TensorMin
	TensorMax
)

type TensorTensorMath[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	O OperTensor
}

func TensorMath[T types.Number](t Tensor[T], oper OperTensor) Tensor[T] {
	return &TensorTensorMath[T]{
		T: t,
		O: oper,
	}
}

func (p *TensorTensorMath[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = []uint{1}
	p.MulIndex = []uint{1}
	op := &number.Scalar[T]{}
	p.Operands = []*number.Scalar[T]{op}

	switch p.O {
	case TensorSum:
		ctx.Push(&operation.Sum[T]{
			Scalar: op,
			Args:   p.T.GetOperands(),
		})
	case TensorAvg:
		ctx.Push(&operation.Avg[T]{
			Scalar: op,
			Args:   p.T.GetOperands(),
		})
	case TensorMin:
		ctx.Push(&operation.Min[T]{
			Scalar: op,
			Args:   p.T.GetOperands(),
		})
	case TensorMax:
		ctx.Push(&operation.Max[T]{
			Scalar: op,
			Args:   p.T.GetOperands(),
		})
	}

	return nil
}
