package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type OperScalar byte

const (
	OpSAdd = OperScalar(iota)
	OpSSub
	OpSMul
	OpSDiv
)

type TensorOpScalar[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	S *operation.Operand[T]
	O OperScalar
}

func NewOpScalar[T types.Number](t Tensor[T], s *operation.Operand[T], o OperScalar) Tensor[T] {
	return &TensorOpScalar[T]{
		T: t,
		S: s,
		O: o,
	}
}

func (p *TensorOpScalar[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	p.Shape = p.T.GetShape()
	p.MulIndex = p.T.GetMulIndex()

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	op := p.T.GetOperands()

	p.Operands = make([]*operation.Operand[T], p.T.GetSize())
	for i := range p.Operands {
		o := &operation.Operand[T]{}
		p.Operands[i] = o
		switch p.O {
		case OpSAdd:
			ctx.Push(&operation.Add[T]{Operand: o, A: op[i], B: p.S})
		case OpSSub:
			ctx.Push(&operation.Sub[T]{Operand: o, A: op[i], B: p.S})
		case OpSMul:
			ctx.Push(&operation.Mul[T]{Operand: o, A: op[i], B: p.S})
		case OpSDiv:
			ctx.Push(&operation.Div[T]{Operand: o, A: op[i], B: p.S})
		}
	}

	return nil
}
