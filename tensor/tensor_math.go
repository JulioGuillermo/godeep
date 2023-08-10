package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/types"
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
	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = []uint{1}
	p.MulIndex = []uint{1}
	op := &operation.Operand[T]{}
	p.Operands = []*operation.Operand[T]{op}

	switch p.O {
	case TensorSum:
		ctx.Push(&operation.Sum[T]{
			Operand: op,
			Args:    p.T.GetOperands(),
		})
	case TensorAvg:
		ctx.Push(&operation.Avg[T]{
			Operand: op,
			Args:    p.T.GetOperands(),
		})
	case TensorMin:
		ctx.Push(&operation.Min[T]{
			Operand: op,
			Args:    p.T.GetOperands(),
		})
	case TensorMax:
		ctx.Push(&operation.Max[T]{
			Operand: op,
			Args:    p.T.GetOperands(),
		})
	}

	return nil
}
