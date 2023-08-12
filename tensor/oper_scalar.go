package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
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
	S *number.Scalar[T]
	O OperScalar
}

func NewOpScalar[T types.Number](t Tensor[T], s *number.Scalar[T], o OperScalar) Tensor[T] {
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

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = p.T.GetShape()
	p.MulIndex = p.T.GetMulIndex()

	op := p.T.GetOperands()

	p.Operands = make([]*number.Scalar[T], p.T.GetSize())
	for i := range p.Operands {
		o := &number.Scalar[T]{}
		p.Operands[i] = o
		switch p.O {
		case OpSAdd:
			ctx.Push(&operation.Add[T]{Scalar: o, A: op[i], B: p.S})
		case OpSSub:
			ctx.Push(&operation.Sub[T]{Scalar: o, A: op[i], B: p.S})
		case OpSMul:
			ctx.Push(&operation.Mul[T]{Scalar: o, A: op[i], B: p.S})
		case OpSDiv:
			ctx.Push(&operation.Div[T]{Scalar: o, A: op[i], B: p.S})
		}
	}

	return nil
}
