package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/errors"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type OperDim byte

const (
	DimSum = OperDim(iota)
	DimAvg
	DimMin
	DimMax
)

type TensorDimMath[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	D uint
	O OperDim
}

func DimMath[T types.Number](t Tensor[T], dim uint, oper OperDim) Tensor[T] {
	return &TensorDimMath[T]{
		T: t,
		D: dim,
		O: oper,
	}
}

func (p *TensorDimMath[T]) BuildGraph(ctx *context.Context) error {
	if p.D >= uint(len(p.T.GetShape())) {
		return errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			p.D,
			len(p.T.GetShape()),
		)
	}
	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = p.T.GetShape()[:p.D+1]
	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*operation.Operand[T], tools.GetDataSize(p.Shape))

	mi := p.T.GetMulIndex()[p.D]
	ops := p.T.GetOperands()
	for i := range p.Operands {
		offset := uint(i) * mi
		o := &operation.Operand[T]{}
		p.Operands[i] = o
		args := ops[offset : offset+mi]

		switch p.O {
		case DimSum:
			ctx.Push(&operation.Sum[T]{
				Operand: o,
				Args:    args,
			})
		case DimAvg:
			ctx.Push(&operation.Avg[T]{
				Operand: o,
				Args:    args,
			})
		case DimMin:
			ctx.Push(&operation.Min[T]{
				Operand: o,
				Args:    args,
			})
		case DimMax:
			ctx.Push(&operation.Max[T]{
				Operand: o,
				Args:    args,
			})
		}
	}

	return nil
}
